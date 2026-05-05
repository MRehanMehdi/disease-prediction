import joblib
import pandas as pd

# ── Load model and symptom list ───────────────────────────────
model = joblib.load('models/best_model.pkl')
symptom_list = joblib.load('models/symptom_list.pkl')


def predict_top3_diseases(selected_symptoms):
    """Returns top 3 predicted diseases with confidence scores."""
    input_vector = [1 if sym in selected_symptoms else 0 for sym in symptom_list]
    input_df = pd.DataFrame([input_vector], columns=symptom_list)
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_
    disease_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return [(disease, round(prob * 100, 2)) for disease, prob in disease_probs[:3]]


def get_disease_info(disease_name, desc_df, prec_df):
    """Returns description and precautions for a disease."""
    desc_row = desc_df[desc_df['Disease'].str.strip() == disease_name]
    description = desc_row['Description'].values[0] if not desc_row.empty else "No description available."
    prec_row = prec_df[prec_df['Disease'].str.strip() == disease_name]
    precautions = []
    if not prec_row.empty:
        for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
            val = prec_row[col].values[0]
            if pd.notna(val) and str(val).strip() != '':
                precautions.append(val.strip())
    return description, precautions


def get_severity_score(selected_symptoms, severity_df):
    """Calculates average severity score and risk level."""
    total = 0
    matched = 0
    for sym in selected_symptoms:
        row = severity_df[severity_df['Symptom'].str.strip() == sym]
        if not row.empty:
            total += row['weight'].values[0]
            matched += 1
    if matched == 0:
        return 0, "Unknown"
    avg = round(total / matched, 2)
    risk = "High" if avg >= 5 else "Medium" if avg >= 3 else "Low"
    return avg, risk


def generate_pdf_report(selected_symptoms, top3, description, precautions, severity_score, risk_level):
    """Generates a PDF report and returns it as bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import cm
    import io

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('title', fontSize=20, fontName='Helvetica-Bold',
                                  textColor=colors.HexColor('#0d6efd'), spaceAfter=5)
    story.append(Paragraph("Disease Prediction Report", title_style))

    sub_style = ParagraphStyle('sub', fontSize=10, textColor=colors.grey, spaceAfter=15)
    story.append(Paragraph("Riphah International University | AI Course Project | 6th Semester BSCS", sub_style))
    story.append(Spacer(1, 0.3*cm))

    heading = ParagraphStyle('h2', fontSize=13, fontName='Helvetica-Bold', spaceAfter=6)
    normal = styles['Normal']

    # Selected Symptoms
    story.append(Paragraph("Selected Symptoms", heading))
    story.append(Paragraph(", ".join(selected_symptoms), normal))
    story.append(Spacer(1, 0.4*cm))

    # Severity
    story.append(Paragraph("Symptom Severity", heading))
    story.append(Paragraph(f"Risk Level: {risk_level} | Severity Score: {severity_score}/7", normal))
    story.append(Spacer(1, 0.4*cm))

    # Top 3 Predictions
    story.append(Paragraph("Top 3 Predicted Diseases", heading))
    pred_data = [['Rank', 'Disease', 'Confidence %']]
    ranks = ['1st - Most Likely', '2nd - Possible', '3rd - Less Likely']
    for rank, (disease, conf) in zip(ranks, top3):
        pred_data.append([rank, disease, f"{conf}%"])

    pred_table = Table(pred_data, colWidths=[4*cm, 9*cm, 3*cm])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d6efd')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 0.4*cm))

    # About Disease
    story.append(Paragraph(f"About: {top3[0][0]}", heading))
    story.append(Paragraph(description, normal))
    story.append(Spacer(1, 0.4*cm))

    # Precautions
    story.append(Paragraph("Recommended Precautions", heading))
    for i, p in enumerate(precautions, 1):
        story.append(Paragraph(f"{i}. {p.capitalize()}", normal))
    story.append(Spacer(1, 0.4*cm))

    # Disclaimer
    disclaimer_style = ParagraphStyle('disc', fontSize=8, textColor=colors.red)
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system for educational purposes only. "
        "It does not replace professional medical advice. Please consult a qualified physician.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer