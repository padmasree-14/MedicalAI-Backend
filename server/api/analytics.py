import os
import json
import time
from datetime import datetime
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException

from server.database.db import db_manager
from server.api.auth import get_current_user

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

MODEL_METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "metrics.json")

@router.get("")
async def get_analytics(current_user: dict = Depends(get_current_user)):
    # 1. Fetch all predictions for the user
    user_predictions = await db_manager.get_predictions(current_user["id"])
    
    total_predictions = len(user_predictions)
    
    # 2. Compute disease distribution
    disease_distribution = defaultdict(int)
    for pred in user_predictions:
        p_class = pred.get("predicted_class", "UNKNOWN").upper()
        disease_distribution[p_class] += 1
        
    # Format distribution for charts
    formatted_distribution = [
        {"name": name, "value": count} for name, count in disease_distribution.items()
    ]
    
    # 3. Monthly analysis (group by month-year or day-month)
    monthly_data = defaultdict(int)
    for pred in user_predictions:
        created_timestamp = pred.get("created_at", time.time())
        # Convert timestamp to human readable Month name
        month_str = datetime.fromtimestamp(created_timestamp).strftime("%b %y")
        monthly_data[month_str] += 1
        
    # Order monthly analysis (newest or chronological)
    # We'll return it as a list of dictionaries sorted chronologically if possible, or just a list
    formatted_monthly = [
        {"month": month, "scans": count} for month, count in monthly_data.items()
    ]
    # Ensure it's not empty for Recharts
    if not formatted_monthly:
        formatted_monthly = [{"month": datetime.now().strftime("%b %y"), "scans": 0}]
    else:
        # Reverse to show chronological order (or sorted)
        formatted_monthly.reverse()

    # 4. Try to load training metrics from the saved training file
    ml_metrics = {}
    if os.path.exists(MODEL_METRICS_PATH):
        try:
            with open(MODEL_METRICS_PATH, "r") as f:
                ml_metrics = json.load(f)
        except Exception as e:
            print(f"Error loading model metrics for analytics: {e}")
            
    # Compile response
    return {
        "stats": {
            "total_predictions": total_predictions,
            "normal_count": disease_distribution.get("NORMAL", 0),
            "pneumonia_count": disease_distribution.get("PNEUMONIA", 0),
            "average_confidence": sum([p.get("confidence", 0) for p in user_predictions]) / total_predictions if total_predictions > 0 else 0.0
        },
        "disease_distribution": formatted_distribution,
        "monthly_analysis": formatted_monthly,
        "ml_metrics": ml_metrics
    }
