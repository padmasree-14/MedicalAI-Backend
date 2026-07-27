import os
import uuid
import shutil
import time
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from fastapi.responses import FileResponse

from server.database.db import db_manager
from server.database.models import PredictionResponse, ReportResponse
from server.api.auth import get_current_user, get_current_user_optional
from server.model.predict import get_predictor
from server.gradcam.gradcam import generate_gradcam
from server.utils.llm import generate_ai_report
from server.utils.pdf_generator import build_patient_pdf

router = APIRouter(prefix="/api", tags=["Predictions"])

# Local folder configurations
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_name: str = Form("Anonymous Patient"),
    current_user: dict = Depends(get_current_user_optional)
):
    # 1. Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload JPG, PNG or BMP."
        )

    # 2. Save uploaded file to uploads directory with unique name
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    input_filepath = os.path.join(UPLOADS_DIR, unique_filename)
    
    try:
        file_bytes = await file.read()
        with open(input_filepath, "wb") as buffer:
            buffer.write(file_bytes)
            buffer.flush()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded image: {e}"
        )

    # 3. Execute Deep Learning model prediction
    try:
        predictor = get_predictor()
        predicted_class, confidence, probabilities = predictor.predict_image(input_filepath, file_bytes=file_bytes)
    except Exception as e:
        # Cleanup uploaded file if prediction fails
        if os.path.exists(input_filepath):
            os.remove(input_filepath)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model prediction failed: {e}"
        )

    # 4. Generate Grad-CAM activation heatmap overlay
    try:
        # The predictor contains model loaded into TF context
        overlaid_path, _ = generate_gradcam(
            model=predictor.model,
            img_path=input_filepath,
            target_layer_name="conv_last",
            output_dir=UPLOADS_DIR
        )
        # Relative filepaths for static asset server
        gradcam_filename = os.path.basename(overlaid_path)
    except Exception as e:
        print(f"Grad-CAM generation warning: {e}")
        gradcam_filename = None

    # 5. Insert prediction into database
    pred_data = {
        "user_id": current_user["id"],
        "filename": file.filename,
        "filepath": input_filepath,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "gradcam_path": overlaid_path if gradcam_filename else None
    }
    
    prediction_record = await db_manager.create_prediction(pred_data)
    
    # 6. Generate AI Clinical Diagnosis Text via LLM
    report_text = await generate_ai_report(predicted_class, confidence, patient_name)
    
    # 7. Generate printable Clinical PDF Report via ReportLab
    pdf_filename = f"report_{prediction_record['id']}.pdf"
    pdf_filepath = os.path.join(REPORTS_DIR, pdf_filename)
    
    try:
        build_patient_pdf(
            output_path=pdf_filepath,
            prediction_data=prediction_record,
            report_data=report_text,
            username=current_user.get("username", "Staff Clinician")
        )
    except Exception as e:
        print(f"PDF creation failure: {e}")
        pdf_filepath = ""

    # 8. Save report record to DB
    report_data = {
        "prediction_id": prediction_record["id"],
        "user_id": current_user["id"],
        "pdf_path": pdf_filepath,
        "report_text": report_text
    }
    await db_manager.create_report(report_data)

    # Log action
    await db_manager.log_action(
        current_user["id"],
        "prediction_run",
        f"Ran prediction on {file.filename} -> Result: {predicted_class} ({confidence*100:.1f}%)"
    )

    return {
        "prediction": {
            "id": prediction_record["id"],
            "filename": prediction_record["filename"],
            "predicted_class": prediction_record["predicted_class"],
            "confidence": prediction_record["confidence"],
            "probabilities": prediction_record["probabilities"],
            "gradcam_url": f"/static/uploads/{gradcam_filename}" if gradcam_filename else None,
            "original_url": f"/static/uploads/{unique_filename}",
            "created_at": prediction_record["created_at"]
        },
        "report": {
            "patient_name": patient_name,
            "report_text": report_text,
            "pdf_url": f"/api/reports/download/{prediction_record['id']}"
        }
    }

@router.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    raw_history = await db_manager.get_predictions(current_user["id"])
    
    # Format URLs for frontend exposure
    formatted_history = []
    for item in raw_history:
        original_filename = os.path.basename(item["filepath"])
        gradcam_url = None
        if item.get("gradcam_path"):
            gradcam_url = f"/static/uploads/{os.path.basename(item['gradcam_path'])}"
            
        formatted_history.append({
            "id": item["id"],
            "filename": item["filename"],
            "predicted_class": item["predicted_class"],
            "confidence": item["confidence"],
            "probabilities": item["probabilities"],
            "original_url": f"/static/uploads/{original_filename}",
            "gradcam_url": gradcam_url,
            "created_at": item["created_at"]
        })
    return formatted_history

@router.delete("/history/{id}")
async def delete_history_item(id: str, current_user: dict = Depends(get_current_user)):
    # 1. Fetch prediction details to get file paths
    pred_item = await db_manager.get_prediction_by_id(id)
    if not pred_item or pred_item["user_id"] != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction history item not found."
        )
        
    # 2. Delete prediction record from DB
    deleted = await db_manager.delete_prediction(id, current_user["id"])
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete prediction record.")

    # 3. Clean up physical image uploads
    try:
        if pred_item.get("filepath") and os.path.exists(pred_item["filepath"]):
            os.remove(pred_item["filepath"])
        if pred_item.get("gradcam_path") and os.path.exists(pred_item["gradcam_path"]):
            os.remove(pred_item["gradcam_path"])
    except Exception as e:
        print(f"File cleanup warning during history delete: {e}")

    # 4. Clean up associated PDF reports
    try:
        report_item = await db_manager.get_report_by_prediction(id, current_user["id"])
        if report_item and report_item.get("pdf_path") and os.path.exists(report_item["pdf_path"]):
            os.remove(report_item["pdf_path"])
    except Exception as e:
        print(f"Report cleanup warning during history delete: {e}")

    # Log action
    await db_manager.log_action(current_user["id"], "prediction_delete", f"Deleted prediction reference ID {id}.")
    
    return {"message": "Prediction item deleted successfully."}

@router.get("/reports/download/{prediction_id}")
async def download_report_pdf(prediction_id: str, current_user: dict = Depends(get_current_user)):
    report_item = await db_manager.get_report_by_prediction(prediction_id, current_user["id"])
    if not report_item or not report_item.get("pdf_path") or not os.path.exists(report_item["pdf_path"]):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report PDF file not found."
        )
        
    return FileResponse(
        path=report_item["pdf_path"],
        filename=os.path.basename(report_item["pdf_path"]),
        media_type="application/pdf"
    )

@router.get("/predictions/{prediction_id}/report")
async def get_prediction_report(prediction_id: str, current_user: dict = Depends(get_current_user)):
    report_item = await db_manager.get_report_by_prediction(prediction_id, current_user["id"])
    if not report_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report text not found."
        )
    pred_item = await db_manager.get_prediction_by_id(prediction_id)
    if not pred_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction metadata not found."
        )
    
    original_filename = os.path.basename(pred_item["filepath"])
    gradcam_url = None
    if pred_item.get("gradcam_path"):
        gradcam_url = f"/static/uploads/{os.path.basename(pred_item['gradcam_path'])}"
        
    return {
        "prediction_id": report_item["prediction_id"],
        "report_text": report_item["report_text"],
        "pdf_url": f"/api/reports/download/{prediction_id}",
        "prediction": {
            "id": pred_item["id"],
            "filename": pred_item["filename"],
            "predicted_class": pred_item["predicted_class"],
            "confidence": pred_item["confidence"],
            "original_url": f"/static/uploads/{original_filename}",
            "gradcam_url": gradcam_url,
            "created_at": pred_item["created_at"]
        }
    }

@router.post("/generate-report")
async def generate_report_post(payload: dict, current_user: dict = Depends(get_current_user)):
    """Explicit report generation API supporting POST as specified in instructions"""
    prediction_id = payload.get("prediction_id")
    patient_name = payload.get("patient_name", "Anonymous Patient")
    
    if not prediction_id:
        raise HTTPException(status_code=400, detail="Missing prediction_id in request body.")
        
    # Get prediction
    pred = await db_manager.get_prediction_by_id(prediction_id)
    if not pred or pred["user_id"] != current_user["id"]:
        raise HTTPException(status_code=404, detail="Prediction item not found.")
        
    # Check if report already exists
    existing_report = await db_manager.get_report_by_prediction(prediction_id, current_user["id"])
    if existing_report:
        return {
            "prediction_id": prediction_id,
            "report_text": existing_report["report_text"],
            "pdf_url": f"/api/reports/download/{prediction_id}"
        }
        
    # Otherwise generate it
    report_text = await generate_ai_report(pred["predicted_class"], pred["confidence"], patient_name)
    
    pdf_filename = f"report_{prediction_id}.pdf"
    pdf_filepath = os.path.join(REPORTS_DIR, pdf_filename)
    
    try:
        build_patient_pdf(
            output_path=pdf_filepath,
            prediction_data=pred,
            report_data=report_text,
            username=current_user.get("username", "Staff Clinician")
        )
    except Exception as e:
        print(f"PDF creation failure: {e}")
        pdf_filepath = ""
        
    report_data = {
        "prediction_id": prediction_id,
        "user_id": current_user["id"],
        "pdf_path": pdf_filepath,
        "report_text": report_text
    }
    await db_manager.create_report(report_data)
    
    return {
        "prediction_id": prediction_id,
        "report_text": report_text,
        "pdf_url": f"/api/reports/download/{prediction_id}"
    }
