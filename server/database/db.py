import os
import time
import asyncio
import sqlite3
import json
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError
from bson.objectid import ObjectId

# Database config
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/medical_ai")
SQLITE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "medical_ai.db")

class DatabaseManager:
    def __init__(self):
        self.use_sqlite = False
        self.mongo_client = None
        self.db = None
        
    async def initialize(self):
        """Tries to connect to MongoDB. If it fails, falls back to SQLite."""
        try:
            print("Attempting to connect to MongoDB...")
            self.mongo_client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=2000)
            # Trigger connection test
            await self.mongo_client.admin.command('ping')
            self.db = self.mongo_client.get_database()
            self.use_sqlite = False
            print("Successfully connected to MongoDB.")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            print("Falling back to local SQLite database...")
            self.use_sqlite = True
            await self._init_sqlite()

    async def _init_sqlite(self):
        """Initializes SQLite tables in a thread-safe pool execution."""
        def run_init():
            conn = sqlite3.connect(SQLITE_DB_PATH)
            c = conn.cursor()
            
            # Users table
            c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password_hash TEXT,
                clinic_name TEXT,
                created_at REAL
            )
            """)
            
            # Predictions table
            c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                filename TEXT,
                filepath TEXT,
                predicted_class TEXT,
                confidence REAL,
                probabilities TEXT,
                gradcam_path TEXT,
                created_at REAL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """)
            
            # Reports table
            c.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                prediction_id TEXT,
                user_id TEXT,
                pdf_path TEXT,
                report_text TEXT,
                created_at REAL,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """)
            
            # Logs table
            c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                action TEXT,
                details TEXT,
                timestamp REAL
            )
            """)
            
            conn.commit()
            conn.close()
            
        await asyncio.to_thread(run_init)
        print(f"Local SQLite database initialized at {SQLITE_DB_PATH}")

    # ================= USER OPERATIONS =================
    async def create_user(self, user_dict):
        """Creates a new user"""
        user_dict = dict(user_dict)
        user_dict["created_at"] = time.time()
        
        if not self.use_sqlite:
            # MongoDB
            res = await self.db.users.insert_one(user_dict)
            user_dict["id"] = str(res.inserted_id)
            user_dict["_id"] = user_dict["id"]
            return user_dict
        else:
            # SQLite
            user_dict["id"] = user_dict.get("id") or str(ObjectId())
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                c = conn.cursor()
                try:
                    c.execute("""
                    INSERT INTO users (id, username, email, password_hash, clinic_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        user_dict["id"],
                        user_dict["username"],
                        user_dict["email"],
                        user_dict["password_hash"],
                        user_dict.get("clinic_name", ""),
                        user_dict["created_at"]
                    ))
                    conn.commit()
                    return True
                except sqlite3.IntegrityError as ie:
                    print(f"SQLite Integrity Error: {ie}")
                    return False
                finally:
                    conn.close()
            
            success = await asyncio.to_thread(run_db)
            if success:
                return user_dict
            return None

    async def get_user_by_email(self, email):
        """Retrieves a user by email"""
        if not self.use_sqlite:
            user = await self.db.users.find_one({"email": email})
            if user:
                user["id"] = str(user["_id"])
                del user["_id"]
            return user
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE email = ?", (email,))
                row = c.fetchone()
                conn.close()
                if row:
                    return dict(row)
                return None
            return await asyncio.to_thread(run_db)

    async def get_user_by_id(self, user_id):
        """Retrieves a user by ID"""
        if not self.use_sqlite:
            try:
                user = await self.db.users.find_one({"_id": ObjectId(user_id)})
                if user:
                    user["id"] = str(user["_id"])
                    del user["_id"]
                return user
            except Exception:
                return None
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
                row = c.fetchone()
                conn.close()
                if row:
                    return dict(row)
                return None
            return await asyncio.to_thread(run_db)

    async def update_user(self, user_id, update_dict):
        """Updates user profile"""
        if not self.use_sqlite:
            try:
                await self.db.users.update_one({"_id": ObjectId(user_id)}, {"$set": update_dict})
                return True
            except Exception:
                return False
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                c = conn.cursor()
                fields = ", ".join([f"{k} = ?" for k in update_dict.keys()])
                values = list(update_dict.values()) + [user_id]
                c.execute(f"UPDATE users SET {fields} WHERE id = ?", values)
                conn.commit()
                conn.close()
                return True
            return await asyncio.to_thread(run_db)

    # ================= PREDICTION OPERATIONS =================
    async def create_prediction(self, pred_dict):
        pred_dict = dict(pred_dict)
        pred_dict["created_at"] = time.time()
        
        if not self.use_sqlite:
            res = await self.db.predictions.insert_one(pred_dict)
            pred_dict["id"] = str(res.inserted_id)
            del pred_dict["_id"]
            return pred_dict
        else:
            pred_dict["id"] = pred_dict.get("id") or str(ObjectId())
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                c = conn.cursor()
                c.execute("""
                INSERT INTO predictions (id, user_id, filename, filepath, predicted_class, confidence, probabilities, gradcam_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred_dict["id"],
                    pred_dict["user_id"],
                    pred_dict["filename"],
                    pred_dict["filepath"],
                    pred_dict["predicted_class"],
                    pred_dict["confidence"],
                    json.dumps(pred_dict["probabilities"]),
                    pred_dict["gradcam_path"],
                    pred_dict["created_at"]
                ))
                conn.commit()
                conn.close()
            await asyncio.to_thread(run_db)
            return pred_dict

    async def get_predictions(self, user_id):
        """Retrieves history of predictions for a specific user"""
        if not self.use_sqlite:
            cursor = self.db.predictions.find({"user_id": user_id}).sort("created_at", -1)
            predictions = []
            async for doc in cursor:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
                predictions.append(doc)
            return predictions
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
                rows = c.fetchall()
                conn.close()
                results = []
                for row in rows:
                    item = dict(row)
                    item["probabilities"] = json.loads(item["probabilities"])
                    results.append(item)
                return results
            return await asyncio.to_thread(run_db)

    async def get_prediction_by_id(self, pred_id):
        if not self.use_sqlite:
            try:
                doc = await self.db.predictions.find_one({"_id": ObjectId(pred_id)})
                if doc:
                    doc["id"] = str(doc["_id"])
                    del doc["_id"]
                return doc
            except Exception:
                return None
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
                row = c.fetchone()
                conn.close()
                if row:
                    item = dict(row)
                    item["probabilities"] = json.loads(item["probabilities"])
                    return item
                return None
            return await asyncio.to_thread(run_db)

    async def delete_prediction(self, pred_id, user_id):
        """Deletes prediction entry from database"""
        if not self.use_sqlite:
            try:
                res = await self.db.predictions.delete_one({"_id": ObjectId(pred_id), "user_id": user_id})
                return res.deleted_count > 0
            except Exception:
                return False
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                c = conn.cursor()
                c.execute("DELETE FROM predictions WHERE id = ? AND user_id = ?", (pred_id, user_id))
                count = c.rowcount
                conn.commit()
                conn.close()
                return count > 0
            return await asyncio.to_thread(run_db)

    # ================= REPORT OPERATIONS =================
    async def create_report(self, report_dict):
        report_dict = dict(report_dict)
        report_dict["created_at"] = time.time()
        
        if not self.use_sqlite:
            res = await self.db.reports.insert_one(report_dict)
            report_dict["id"] = str(res.inserted_id)
            del report_dict["_id"]
            return report_dict
        else:
            report_dict["id"] = report_dict.get("id") or str(ObjectId())
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                c = conn.cursor()
                c.execute("""
                INSERT INTO reports (id, prediction_id, user_id, pdf_path, report_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    report_dict["id"],
                    report_dict["prediction_id"],
                    report_dict["user_id"],
                    report_dict["pdf_path"],
                    json.dumps(report_dict["report_text"]),
                    report_dict["created_at"]
                ))
                conn.commit()
                conn.close()
            await asyncio.to_thread(run_db)
            return report_dict

    async def get_report_by_prediction(self, pred_id, user_id):
        if not self.use_sqlite:
            doc = await self.db.reports.find_one({"prediction_id": pred_id, "user_id": user_id})
            if doc:
                doc["id"] = str(doc["_id"])
                del doc["_id"]
            return doc
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT * FROM reports WHERE prediction_id = ? AND user_id = ?", (pred_id, user_id))
                row = c.fetchone()
                conn.close()
                if row:
                    item = dict(row)
                    item["report_text"] = json.loads(item["report_text"])
                    return item
                return None
            return await asyncio.to_thread(run_db)

    # ================= LOG SYSTEM =================
    async def log_action(self, user_id, action, details):
        log_entry = {
            "id": str(ObjectId()),
            "user_id": user_id,
            "action": action,
            "details": details,
            "timestamp": time.time()
        }
        if not self.use_sqlite:
            try:
                await self.db.logs.insert_one(log_entry)
            except Exception as e:
                print(f"Failed to save log to MongoDB: {e}")
        else:
            def run_db():
                conn = sqlite3.connect(SQLITE_DB_PATH)
                c = conn.cursor()
                c.execute("""
                INSERT INTO logs (id, user_id, action, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """, (log_entry["id"], user_id, action, details, log_entry["timestamp"]))
                conn.commit()
                conn.close()
            await asyncio.to_thread(run_db)

# Global DB manager singleton
db_manager = DatabaseManager()
