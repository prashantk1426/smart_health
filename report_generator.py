import io
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(prediction, user):
    """
    Generates a beautifully structured PDF health report.
    Returns: Bytes of the PDF file.
    """
    buffer = io.BytesIO()
    
    # Page setup
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY_COLOR = colors.HexColor('#0d1b2a')    # Dark slate blue
    SECONDARY_COLOR = colors.HexColor('#1b263b')  # Dark blue-gray
    ACCENT_COLOR = colors.HexColor('#ef4444') if prediction.result == "Positive" else colors.HexColor('#22c55e')
    BG_LIGHT = colors.HexColor('#f8fafc')
    TEXT_COLOR = colors.HexColor('#1e293b')       # Dark gray
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=PRIMARY_COLOR,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=20
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=PRIMARY_COLOR,
        spaceBefore=10,
        spaceAfter=10,
        borderColor=colors.HexColor('#cbd5e1'),
        borderWidth=0.5,
        borderPadding=4
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=TEXT_COLOR,
        leading=14
    )
    
    bold_style = ParagraphStyle(
        'Bold_Custom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    result_style = ParagraphStyle(
        'Result_Custom',
        parent=body_style,
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=ACCENT_COLOR
    )

    # 1. Header (Banner)
    story.append(Paragraph("SMART HEALTH CLINICAL ASSESSMENT REPORT", title_style))
    created_time = prediction.created_at.strftime('%Y-%m-%d %I:%M %p') if isinstance(prediction.created_at, datetime) else str(prediction.created_at)
    story.append(Paragraph(f"Generated on: {created_time}  |  Report ID: CR-{prediction.id:06d}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # 2. Patient Demographics Block
    story.append(Paragraph("Patient Information", h2_style))
    patient_data = [
        [
            Paragraph("Full Name:", bold_style), Paragraph(user.full_name, body_style),
            Paragraph("Age / Gender:", bold_style), Paragraph(f"{user.age or 'N/A'} yrs / {user.gender or 'N/A'}", body_style)
        ],
        [
            Paragraph("Username:", bold_style), Paragraph(user.username, body_style),
            Paragraph("Email Address:", bold_style), Paragraph(user.email, body_style)
        ]
    ]
    t_patient = Table(patient_data, colWidths=[1.2*inch, 2.3*inch, 1.2*inch, 2.3*inch])
    t_patient.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#f1f5f9')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_patient)
    story.append(Spacer(1, 20))
    
    # 3. Assessment Results Block
    story.append(Paragraph("Clinical Risk Assessment Findings", h2_style))
    result_text = "HIGH RISK / POSITIVE INDICATORS DETECTED" if prediction.result == "Positive" else "LOW RISK / NEGATIVE INDICATORS DETECTED"
    assessment_data = [
        [Paragraph("Target Disease:", bold_style), Paragraph(prediction.disease, bold_style)],
        [Paragraph("Diagnostic Result:", bold_style), Paragraph(result_text, result_style)],
        [Paragraph("Model Confidence:", bold_style), Paragraph(f"{prediction.confidence:.2f}%", bold_style)],
        [Paragraph("Risk Severity Level:", bold_style), Paragraph(prediction.risk_level, ParagraphStyle('Risk', parent=bold_style, textColor=ACCENT_COLOR))]
    ]
    t_assess = Table(assessment_data, colWidths=[2.0*inch, 5.0*inch])
    t_assess.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1.0, SECONDARY_COLOR),
    ]))
    story.append(t_assess)
    story.append(Spacer(1, 20))
    
    # 4. Input Features Table
    story.append(Paragraph("Patient Clinical Input Parameters Analyzed", h2_style))
    
    # Parse inputs from JSON string
    try:
        inputs = json.loads(prediction.input_data)
    except Exception:
        inputs = {}
        
    input_rows = []
    # Display in a 2-column layout (Label and Value)
    idx = 0
    temp_row = []
    for label, val in inputs.items():
        # Clean label strings
        display_label = label.replace('_', ' ').title()
        
        # Safe numeric formatting
        try:
            formatted_val = f"{float(val):.2f}" if '.' in str(val) else str(val)
        except ValueError:
            formatted_val = str(val)
            
        temp_row.extend([Paragraph(display_label, bold_style), Paragraph(formatted_val, body_style)])
        idx += 1
        if idx % 2 == 0:
            input_rows.append(temp_row)
            temp_row = []
            
    # Add trailing odd row if exists
    if temp_row:
        temp_row.extend(["", ""])
        input_rows.append(temp_row)
        
    if not input_rows:
        input_rows = [[Paragraph("No clinical inputs captured.", body_style), "", "", ""]]
        
    t_inputs = Table(input_rows, colWidths=[1.8*inch, 1.7*inch, 1.8*inch, 1.7*inch])
    t_inputs.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, BG_LIGHT])
    ]))
    story.append(t_inputs)
    story.append(Spacer(1, 20))
    
    # 5. Recommendations & Disclaimer
    story.append(Paragraph("Clinical Recommendations & Guidance", h2_style))
    if prediction.result == "Positive":
        rec_text = "<b>Action Required:</b> Based on the machine learning risk assessment, positive clinical markers were identified. It is highly recommended that you schedule a consultation with a qualified medical professional or specialist immediately. Share this digital report and inputs with them for comparison and comprehensive diagnosis."
    else:
        rec_text = "<b>Action Guidance:</b> No high-risk clinical markers were identified. Maintain a healthy lifestyle, regular cardiovascular exercises, healthy eating habits, and check your clinical parameters on an annual basis to ensure continuous robust health."
        
    story.append(Paragraph(rec_text, body_style))
    story.append(Spacer(1, 15))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=body_style,
        fontSize=8,
        textColor=colors.HexColor('#64748b'),
        leading=11
    )
    story.append(Paragraph("<b>Disclaimer:</b> This diagnostic assessment is generated using a computational machine learning model trained on general patient cohorts. It is not an official clinical diagnosis and does not substitute for professional medical consultation, diagnostic tests, or clinical treatment plans. Please consult your physician for official diagnostics.", disclaimer_style))
    
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
