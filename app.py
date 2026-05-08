from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io

from utils import (
    predict_top3_diseases,
    predict_nn_top3,
    predict_all_models,
    get_disease_info,
    get_severity_score,
    generate_pdf_report,
    extract_symptoms_from_text,
    extract_symptoms_from_document,
)

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)

# ── Load CSV data once at startup ─────────────────────────────
desc_df     = pd.read_csv('data/symptom_Description.csv')
prec_df     = pd.read_csv('data/symptom_precaution.csv')
severity_df = pd.read_csv('data/Symptom-severity.csv')


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main HTML interface."""
    return render_template('medpredict_ui.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives symptom text from the HTML frontend.
    Runs RF + NN models from utils.py (unchanged logic).
    Returns JSON with predictions.
    """
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No input provided'}), 400

    # Extract symptoms using existing utils.py function
    symptoms = extract_symptoms_from_text(text)

    if len(symptoms) < 2:
        return jsonify({
            'error': 'not_enough_symptoms',
            'symptoms_found': symptoms,
            'count': len(symptoms)
        })

    # Run all models — same functions from utils.py, zero changes
    top3_rf         = predict_top3_diseases(symptoms)   # returns [(disease, conf), ...]
    top3_nn         = predict_nn_top3(symptoms)          # returns [(disease, conf), ...]
    all_model_res   = predict_all_models(symptoms)       # returns dict
    sev_score, risk = get_severity_score(symptoms, severity_df)
    top_disease     = top3_rf[0][0]
    desc, precs     = get_disease_info(top_disease, desc_df, prec_df)

    return jsonify({
        'rf':       [[d, c] for d, c in top3_rf],
        'nn':       [[d, c] for d, c in top3_nn],
        'sev':      risk,
        'score':    sev_score,
        'top':      top_disease,
        'desc':     desc,
        'precs':    precs,
        'symptoms': symptoms
    })


@app.route('/predict-doc', methods=['POST'])
def predict_doc():
    """
    Receives an uploaded file (PDF/image) + optional extra text.
    Extracts symptoms from document using utils.py function.
    Returns JSON with predictions.
    """
    file  = request.files.get('file')
    extra = request.form.get('extra', '').strip()

    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Extract symptoms from document using existing utils.py function
    found_symptoms, preview_text = extract_symptoms_from_document(file)

    # Also extract from any extra text typed by user
    if extra:
        for s in extract_symptoms_from_text(extra):
            if s not in found_symptoms:
                found_symptoms.append(s)

    if len(found_symptoms) < 2:
        return jsonify({
            'error': 'not_enough_symptoms',
            'symptoms_found': found_symptoms,
            'count': len(found_symptoms),
            'preview': preview_text if isinstance(preview_text, str) else ''
        })

    # Run all models — same functions from utils.py, zero changes
    top3_rf         = predict_top3_diseases(found_symptoms)
    top3_nn         = predict_nn_top3(found_symptoms)
    all_model_res   = predict_all_models(found_symptoms)
    sev_score, risk = get_severity_score(found_symptoms, severity_df)
    top_disease     = top3_rf[0][0]
    desc, precs     = get_disease_info(top_disease, desc_df, prec_df)

    return jsonify({
        'rf':       [[d, c] for d, c in top3_rf],
        'nn':       [[d, c] for d, c in top3_nn],
        'sev':      risk,
        'score':    sev_score,
        'top':      top_disease,
        'desc':     desc,
        'precs':    precs,
        'symptoms': found_symptoms,
        'preview':  preview_text if isinstance(preview_text, str) else ''
    })


@app.route('/export-pdf', methods=['POST'])
def export_pdf():
    try:
        data = request.get_json()

        symptoms    = data.get('symptoms', [])
        rf_raw      = data.get('rf', [])
        nn_raw      = data.get('nn', [])
        sev_score   = data.get('score', 0)
        risk        = data.get('sev', 'Unknown')
        desc        = data.get('desc', '')
        precs       = data.get('precs', [])
        top_disease = data.get('top', 'Disease')

        top3_rf = [(row[0], row[1]) for row in rf_raw]
        top3_nn = [(row[0], row[1]) for row in nn_raw]

        all_model_results = {
            '🤖 Random Forest': {
                'disease':    top3_rf[0][0] if top3_rf else '',
                'confidence': top3_rf[0][1] if top3_rf else 0
            },
            '🧠 Neural Network': {
                'disease':    top3_nn[0][0] if top3_nn else '',
                'confidence': top3_nn[0][1] if top3_nn else 0
            }
        }

        pdf_buffer = generate_pdf_report(
            symptoms,
            top3_rf,
            top3_nn,
            all_model_results,
            desc,
            precs,
            sev_score,
            risk
        )

        filename = f"MedPredict_{top_disease.replace(' ', '_')}.pdf"

        # Fixed: pass pdf_buffer directly, no double BytesIO wrapping
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        import traceback
        print("PDF GENERATION ERROR:")
        print(traceback.format_exc())
        return jsonify({'error': f'PDF generation failed: {str(e)}'}), 500

# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 50)
    print("  MedPredict AI — Flask Server")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)