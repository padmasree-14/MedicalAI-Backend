import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.pdfgen import canvas

class ClinicalReportCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def draw_decorations(self):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header line
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(54, 750, 558, 750)
        self.drawString(54, 755, "Advanced AI Medical Intelligence Platform | Diagnostic Center")
        
        # Footer line
        self.line(54, 50, 558, 50)
        self.drawString(54, 38, "CONFIDENTIAL - For Clinical Review Only")
        self.drawRightString(558, 38, "SYSTEM GENERATED CLINICAL DOCUMENT")
        self.restoreState()

def build_patient_pdf(output_path, prediction_data, report_data, username="Staff Clinician"):
    """
    Builds a professional clinical PDF report.
    Args:
        output_path (str): Filepath to save the PDF.
        prediction_data (dict): Prediction details (filename, predicted_class, confidence, probabilities, etc.)
        report_data (dict): LLM summary details (explanation, causes, recommendations, etc.)
        username (str): Active doctor/clinician name.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom ReportLab ParagraphStyles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=15
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748B"),
        spaceBefore=10
    )

    story = []
    
    # Document Header
    story.append(Paragraph("CLINICAL DIAGNOSTIC REPORT", title_style))
    story.append(Spacer(1, 10))
    
    # Patient & Metadata Table
    formatted_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(prediction_data.get("created_at", time.time())))
    
    meta_data = [
        [
            Paragraph("<b>Patient Ref:</b> Anonymous Patient", body_style),
            Paragraph(f"<b>Date:</b> {formatted_date}", body_style)
        ],
        [
            Paragraph(f"<b>Diagnostician:</b> {username}", body_style),
            Paragraph(f"<b>Radiology Image:</b> {prediction_data.get('filename')}", body_style)
        ],
        [
            Paragraph(f"<b>Finding:</b> <font color='#2563EB'><b>{prediction_data.get('predicted_class')}</b></font>", body_style),
            Paragraph(f"<b>Confidence:</b> {prediction_data.get('confidence')*100:.2f}%", body_style)
        ]
    ]
    
    meta_table = Table(meta_data, colWidths=[250, 254])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # Image Section: Original vs Grad-CAM
    # We display original image and overlaid Grad-CAM side-by-side
    orig_path = prediction_data.get("filepath")
    gradcam_path = prediction_data.get("gradcam_path")
    
    images_table_data = []
    col_widths = []
    
    # Verify image paths exist before attempting to insert them in the PDF
    img_row = []
    label_row = []
    
    if orig_path and os.path.exists(orig_path):
        try:
            # Resize image to fit nicely (width ~200, aspect 1:1)
            img1 = Image(orig_path, width=200, height=200)
            img_row.append(img1)
            label_row.append(Paragraph("<b>Input Grayscale Radiography</b>", body_style))
            col_widths.append(250)
        except Exception as e:
            print(f"Error drawing original image in PDF: {e}")
            
    if gradcam_path and os.path.exists(gradcam_path):
        try:
            img2 = Image(gradcam_path, width=200, height=200)
            img_row.append(img2)
            label_row.append(Paragraph("<b>Grad-CAM Visual Activation</b>", body_style))
            col_widths.append(254)
        except Exception as e:
            print(f"Error drawing Grad-CAM image in PDF: {e}")
            
    if img_row:
        images_table_data.append(img_row)
        images_table_data.append(label_row)
        images_table = Table(images_table_data, colWidths=col_widths)
        images_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,0), 5),
            ('TOPPADDING', (0,1), (-1,1), 5),
        ]))
        
        # Keep images together to prevent orphan image pages
        story.append(KeepTogether([
            Paragraph("Diagnostic Visualizations", section_title_style),
            images_table,
            Spacer(1, 15)
        ]))
        
    # Clinical Explanation
    story.append(Paragraph("Clinical Opinion & Summary", section_title_style))
    story.append(Paragraph(report_data.get("patient_summary", ""), body_style))
    story.append(Paragraph(report_data.get("clinical_explanation", ""), body_style))
    story.append(Spacer(1, 10))
    
    # Differential Diagnosis / Possible Causes
    story.append(Paragraph("Differential Diagnostics (Possible Etiologies)", section_title_style))
    for cause in report_data.get("possible_causes", []):
        story.append(Paragraph(f"&bull; {cause}", bullet_style))
    story.append(Spacer(1, 10))
    
    # Recommendations
    story.append(Paragraph("Clinical Recommendations", section_title_style))
    for rec in report_data.get("recommendations", []):
        story.append(Paragraph(f"&bull; {rec}", bullet_style))
    story.append(Spacer(1, 10))
    
    # Lifestyle Suggestions
    story.append(Paragraph("Care Instructions & Lifestyle Interventions", section_title_style))
    for sugg in report_data.get("lifestyle_suggestions", []):
        story.append(Paragraph(f"&bull; {sugg}", bullet_style))
    story.append(Spacer(1, 10))
    
    # Disclaimer
    story.append(Paragraph("Medical Disclaimer", section_title_style))
    story.append(Paragraph(report_data.get("disclaimer", ""), disclaimer_style))
    
    # Build document
    def on_page_trigger(canvas, doc):
        canvas.draw_decorations()
        
    doc.build(story, canvasmaker=ClinicalReportCanvas)
    print(f"Patient Diagnostic PDF saved to {output_path}")
