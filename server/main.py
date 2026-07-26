import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from server.database.db import db_manager
from server.api.auth import router as auth_router, get_current_user
from server.api.predictions import router as pred_router
from server.api.analytics import router as analytics_router

# Ensure training modules are importable from parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(BASE_DIR, "server")
UPLOADS_DIR = os.path.join(SERVER_DIR, "uploads")
REPORTS_DIR = os.path.join(SERVER_DIR, "reports")
METRICS_DIR = os.path.join(SERVER_DIR, "static", "metrics")
MODEL_PATH = os.path.join(SERVER_DIR, "model", "model.keras")

# Create local directories
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize database connection (MongoDB with local SQLite fallback)
    await db_manager.initialize()
    print("Backend service initialized successfully.")
    yield
    # Shutdown logic (if any)
    if db_manager.mongo_client:
        db_manager.mongo_client.close()
        print("MongoDB connection closed.")

app = FastAPI(
    title="Advanced AI Medical Intelligence Platform API",
    description="Core backend services for image predictions, Grad-CAM, and clinical report generations.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware (permitting frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to client URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse
from fastapi.requests import Request

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Global Exception caught: {exc}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

# Mount static asset server to expose uploaded radiographies, Grad-CAM heatmaps, and ML evaluation curves
app.mount("/static/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")
app.mount("/static/metrics", StaticFiles(directory=METRICS_DIR), name="metrics")

# Register API Routers
app.include_router(auth_router)
app.include_router(pred_router)
app.include_router(analytics_router)

# Root route
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Advanced AI Medical Intelligence Platform Backend API",
        "documentation": "/docs"
    }

# ================= ROOT LEVEL ALIASES (Matching exact user specs) =================
# Since the prompt specifies root URLs like POST /register, POST /login, POST /predict,
# we map them to the corresponding API functions directly for absolute compatibility.

@app.post("/register", tags=["Root Aliases"])
async def root_register(payload: dict):
    # Proxy registration to DB directly
    from server.database.models import UserRegister
    from server.api.auth import register as auth_register
    try:
        user_in = UserRegister(**payload)
        return await auth_register(user_in)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login", tags=["Root Aliases"])
async def root_login(payload: dict):
    from server.database.models import UserLogin
    from server.api.auth import login as auth_login
    try:
        credentials = UserLogin(**payload)
        return await auth_login(credentials)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))

from server.api.auth import router as auth_router, get_current_user, get_current_user_optional

@app.post("/predict", tags=["Root Aliases"])
async def root_predict(
    file: UploadFile = File(...),
    patient_name: str = Form("Anonymous Patient"),
    current_user: dict = Depends(get_current_user_optional)
):
    from server.api.predictions import predict as pred_predict
    return await pred_predict(file, patient_name, current_user)

@app.post("/generate-report", tags=["Root Aliases"])
async def root_generate_report(payload: dict, current_user: dict = Depends(get_current_user)):
    from server.api.predictions import generate_report_post
    return await generate_report_post(payload, current_user)

@app.get("/history", tags=["Root Aliases"])
async def root_history(current_user: dict = Depends(get_current_user)):
    from server.api.predictions import get_history as pred_history
    return await pred_history(current_user)

@app.get("/analytics", tags=["Root Aliases"])
async def root_analytics(current_user: dict = Depends(get_current_user)):
    from server.api.analytics import get_analytics as anal_analytics
    return await anal_analytics(current_user)

@app.get("/profile", tags=["Root Aliases"])
async def root_profile(current_user: dict = Depends(get_current_user)):
    from server.api.auth import get_profile as auth_profile
    return await auth_profile(current_user)

@app.delete("/history/{id}", tags=["Root Aliases"])
async def root_delete_history(id: str, current_user: dict = Depends(get_current_user)):
    from server.api.predictions import delete_history_item
    return await delete_history_item(id, current_user)

# Run direct execution
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
