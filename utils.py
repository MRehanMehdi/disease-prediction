import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from rapidfuzz import process, fuzz
import io
import os
import platform

# ══════════════════════════════════════════════════════════════
# SYMPTOM ALIAS DICTIONARY
# ══════════════════════════════════════════════════════════════
SYMPTOM_ALIASES = {

    # ── Fever ─────────────────────────────────────────────────
    "high temperature":             "high_fever",
    "very high fever":              "high_fever",
    "feverish":                     "high_fever",
    "running a fever":              "high_fever",
    "burning up":                   "high_fever",
    "slight fever":                 "mild_fever",
    "low grade fever":              "mild_fever",
    "low fever":                    "mild_fever",
    "fever":                        "high_fever",

    # ── Body pain / weakness ──────────────────────────────────
    "body pain":                    "muscle_pain",
    "body ache":                    "muscle_pain",
    "body hurts":                   "muscle_pain",
    "muscle ache":                  "muscle_pain",
    "whole body hurts":             "muscle_pain",
    "my body hurts":                "muscle_pain",
    "aching all over":              "muscle_pain",
    "general weakness":             "fatigue",
    "feeling weak":                 "fatigue",
    "extreme tiredness":            "fatigue",
    "tired all the time":           "fatigue",
    "always tired":                 "fatigue",
    "no energy":                    "fatigue",
    "weakness":                     "fatigue",
    "feel weak":                    "fatigue",
    "physically weak":              "fatigue",

    # ── Respiratory ───────────────────────────────────────────
    "trouble breathing":            "breathlessness",
    "difficulty breathing":         "breathlessness",
    "shortness of breath":          "breathlessness",
    "cant breathe":                 "breathlessness",
    "can't breathe":                "breathlessness",
    "hard to breathe":              "breathlessness",
    "out of breath":                "breathlessness",
    "breathing difficulty":         "breathlessness",
    "chest tightness":              "breathlessness",
    "wheezing":                      "breathlessness",
    "chest discomfort":             "chest_pain",
    "heart pain":                   "chest_pain",
    "tightness in chest":           "breathlessness",
    "pressure in chest":            "chest_pain",

    # ── Cough / cold ──────────────────────────────────────────
    "runny nose":                   "runny_nose",
    "running nose":                 "runny_nose",
    "nose running":                 "runny_nose",
    "sneezing a lot":               "continuous_sneezing",
    "constant sneezing":            "continuous_sneezing",
    "sneezing continuously":        "continuous_sneezing",
    "dry cough":                    "cough",
    "persistent cough":             "cough",
    "coughing a lot":               "cough",
    "keeps coughing":               "cough",
    "mucus in cough":               "mucoid_sputum",
    "phlegm":                       "phlegm",
    "coughing blood":               "blood_in_sputum",
    "blood in cough":               "blood_in_sputum",
    "sinus pressure":               "sinus_pressure",
    "nasal congestion":             "congestion",
    "blocked nose":                 "congestion",
    "stuffy nose":                  "congestion",

    # ── Urinary ───────────────────────────────────────────────
    "frequent urination":           "polyuria",
    "peeing a lot":                 "polyuria",
    "urinating frequently":         "polyuria",
    "go to bathroom frequently":    "polyuria",
    "bathroom frequently":          "polyuria",
    "urinating too much":           "polyuria",
    "pee frequently":               "polyuria",
    "passing urine frequently":     "polyuria",
    "burning when peeing":          "burning_micturition",
    "painful urination":            "burning_micturition",
    "burning urination":            "burning_micturition",
    "pain when urinating":          "burning_micturition",
    "burn when i pee":              "burning_micturition",
    "bad smell urine":              "foul_smell_of urine",
    "urine smells bad":             "foul_smell_of urine",
    "foul urine":                   "foul_smell_of urine",
    "smelly urine":                 "foul_smell_of urine",
    "dark colored urine":           "dark_urine",
    "urine is dark":                "dark_urine",
    "brown urine":                  "dark_urine",
    "yellow urine":                 "yellow_urine",
    "bladder discomfort":           "bladder_discomfort",
    "urge to urinate":              "continuous_feel_of_urine",
    "always feel like peeing":      "continuous_feel_of_urine",

    # ── Skin ──────────────────────────────────────────────────
    "itchy skin":                   "itching",
    "skin itching":                 "itching",
    "skin is itchy":                "itching",
    "body itching":                 "itching",
    "red rash":                     "skin_rash",
    "rash on skin":                 "skin_rash",
    "rashes":                       "skin_rash",
    "skin rash":                    "skin_rash",
    "rash":                         "skin_rash",
    "red spots":                    "red_spots_over_body",
    "spots on body":                "red_spots_over_body",
    "red spots on skin":            "red_spots_over_body",
    "skin peeling":                 "skin_peeling",
    "peeling skin":                 "skin_peeling",
    "pus pimples":                  "pus_filled_pimples",
    "pimples with pus":             "pus_filled_pimples",
    "skin blister":                 "blister",
    "blisters":                     "blister",
    "yellow skin":                  "yellowish_skin",
    "skin turning yellow":          "yellowish_skin",
    "skin is yellow":               "yellowish_skin",
    "jaundice":                     "yellowish_skin",
    "blackheads":                   "blackheads",
    "bruising":                     "bruising",
    "bruises":                      "bruising",
    "nodal skin eruptions":         "nodal_skin_eruptions",

    # ── Eyes ──────────────────────────────────────────────────
    "blurred vision":               "blurred_and_distorted_vision",
    "blurry vision":                "blurred_and_distorted_vision",
    "blurry eyesight":              "blurred_and_distorted_vision",
    "vision is blurry":             "blurred_and_distorted_vision",
    "cant see clearly":             "blurred_and_distorted_vision",
    "can't see clearly":            "blurred_and_distorted_vision",
    "eye redness":                  "redness_of_eyes",
    "red eyes":                     "redness_of_eyes",
    "eyes are red":                 "redness_of_eyes",
    "watery eyes":                  "watering_from_eyes",
    "eyes watering":                "watering_from_eyes",
    "tears from eyes":              "watering_from_eyes",
    "yellow eyes":                  "yellowing_of_eyes",
    "eyes turning yellow":          "yellowing_of_eyes",
    "pain behind eyes":             "pain_behind_the_eyes",
    "pain behind the eyes":         "pain_behind_the_eyes",
    "eye pain":                     "pain_behind_the_eyes",
    "pressure behind eyes":         "pain_behind_the_eyes",
    "sunken eyes":                  "sunken_eyes",
    "puffy eyes":                   "puffy_face_and_eyes",

    # ── Gastro / digestion ────────────────────────────────────
    "stomach ache":                 "stomach_pain",
    "stomach hurts":                "stomach_pain",
    "pain in stomach":              "stomach_pain",
    "belly ache":                   "belly_pain",
    "belly pain":                   "belly_pain",
    "abdominal pain":               "abdominal_pain",
    "abdominal cramps":             "abdominal_pain",
    "pain in abdomen":              "abdominal_pain",
    "loss of appetite":             "loss_of_appetite",
    "not hungry":                   "loss_of_appetite",
    "no appetite":                  "loss_of_appetite",
    "dont feel like eating":        "loss_of_appetite",
    "don't feel like eating":       "loss_of_appetite",
    "appetite is low":              "loss_of_appetite",
    "feeling like vomiting":        "nausea",
    "feel nauseous":                "nausea",
    "want to vomit":                "nausea",
    "queasy":                       "nausea",
    "feeling sick":                 "nausea",
    "throwing up":                  "vomiting",
    "threw up":                     "vomiting",
    "puking":                       "vomiting",
    "vomited":                      "vomiting",
    "loose motion":                 "diarrhoea",
    "loose motions":                "diarrhoea",
    "watery stool":                 "diarrhoea",
    "diarrhea":                     "diarrhoea",
    "loose stools":                 "diarrhoea",
    "runny stool":                  "diarrhoea",
    "acid reflux":                  "acidity",
    "heartburn":                    "acidity",
    "sour stomach":                 "acidity",
    "bloating":                     "distention_of_abdomen",
    "stomach bloating":             "distention_of_abdomen",
    "stomach is bloated":           "distention_of_abdomen",
    "gas":                          "passage_of_gases",
    "passing gas":                  "passage_of_gases",
    "flatulence":                   "passage_of_gases",
    "constipated":                  "constipation",
    "cant pass stool":              "constipation",
    "can't pass stool":             "constipation",
    "indigestion":                  "indigestion",
    "upset stomach":                "indigestion",
    "stomach bleeding":             "stomach_bleeding",
    "blood in stool":               "bloody_stool",
    "bloody stool":                 "bloody_stool",

    # ── Neurological ──────────────────────────────────────────
    "head spinning":                "dizziness",
    "feeling dizzy":                "dizziness",
    "dizzy":                        "dizziness",
    "lightheaded":                  "dizziness",
    "vertigo":                      "spinning_movements",
    "spinning sensation":           "spinning_movements",
    "room is spinning":             "spinning_movements",
    "cant concentrate":             "lack_of_concentration",
    "can't concentrate":            "lack_of_concentration",
    "hard to focus":                "lack_of_concentration",
    "difficulty concentrating":     "lack_of_concentration",
    "confusion":                    "altered_sensorium",
    "confused":                     "altered_sensorium",
    "disoriented":                  "altered_sensorium",
    "slurred speech":               "slurred_speech",
    "speech problems":              "slurred_speech",
    "stiff neck":                   "stiff_neck",
    "neck stiffness":               "stiff_neck",
    "neck is stiff":                "stiff_neck",
    "unsteady":                     "unsteadiness",
    "loss of balance":              "loss_of_balance",
    "cant balance":                 "loss_of_balance",
    "can't balance":                "loss_of_balance",
    "one side weakness":            "weakness_of_one_body_side",
    "one side of body weak":        "weakness_of_one_body_side",

    # ── Head ──────────────────────────────────────────────────
    "severe headache":              "headache",
    "head pain":                    "headache",
    "migraine":                     "headache",
    "head is pounding":             "headache",
    "throbbing head":               "headache",

    # ── Weight / appetite ─────────────────────────────────────
    "losing weight":                "weight_loss",
    "lost weight":                  "weight_loss",
    "weight is dropping":           "weight_loss",
    "gained weight":                "weight_gain",
    "putting on weight":            "weight_gain",
    "always hungry":                "excessive_hunger",
    "very hungry":                  "excessive_hunger",
    "increased hunger":             "excessive_hunger",
    "hungry all the time":          "excessive_hunger",
    "increased appetite":           "increased_appetite",
    "eating more than usual":       "increased_appetite",

    # ── Thirst / dehydration ──────────────────────────────────
    "very thirsty":                 "dehydration",
    "thirsty all the time":         "dehydration",
    "always thirsty":               "dehydration",
    "extreme thirst":               "dehydration",
    "feeling dehydrated":           "dehydration",
    "dehydrated":                   "dehydration",
    "not drinking enough":          "dehydration",

    # ── Heart / circulation ───────────────────────────────────
    "fast heartbeat":               "fast_heart_rate",
    "heart racing":                 "fast_heart_rate",
    "rapid heartbeat":              "fast_heart_rate",
    "heart is beating fast":        "fast_heart_rate",
    "heart pounding":               "palpitations",
    "heart fluttering":             "palpitations",
    "irregular heartbeat":          "palpitations",
    "swollen legs":                 "swollen_legs",
    "legs are swollen":             "swollen_legs",
    "swollen feet":                 "swollen_extremeties",
    "swollen hands":                "swollen_extremeties",
    "swollen ankles":               "swollen_extremeties",
    "veins popping":                "prominent_veins_on_calf",

    # ── Lymph / immune ────────────────────────────────────────
    "swollen glands":               "swelled_lymph_nodes",
    "swollen lymph nodes":          "swelled_lymph_nodes",
    "glands are swollen":           "swelled_lymph_nodes",

    # ── Mood / mental ─────────────────────────────────────────
    "feeling anxious":              "anxiety",
    "nervous":                      "anxiety",
    "anxious":                      "anxiety",
    "feeling depressed":            "depression",
    "depressed":                    "depression",
    "feeling sad":                  "depression",
    "mood swings":                  "mood_swings",
    "irritable":                    "irritability",
    "easily irritated":             "irritability",
    "restless":                     "restlessness",
    "cant sit still":               "restlessness",
    "can't sit still":              "restlessness",
    "feeling restless":             "restlessness",
    "feeling lethargic":            "lethargy",
    "lethargic":                    "lethargy",
    "sluggish":                     "lethargy",

    # ── Throat / mouth ────────────────────────────────────────
    "sore throat":                  "throat_irritation",
    "throat pain":                  "throat_irritation",
    "throat hurts":                 "throat_irritation",
    "irritated throat":             "throat_irritation",
    "mouth ulcers":                 "ulcers_on_tongue",
    "tongue sores":                 "ulcers_on_tongue",
    "sores in mouth":               "ulcers_on_tongue",
    "patches in throat":            "patches_in_throat",
    "white patches throat":         "patches_in_throat",
    "loss of smell":                "loss_of_smell",
    "cant smell":                   "loss_of_smell",
    "can't smell":                  "loss_of_smell",
    "no sense of smell":            "loss_of_smell",

    # ── Joint / limb pain ─────────────────────────────────────
    "joint pain":                   "joint_pain",
    "joints hurt":                  "joint_pain",
    "joints aching":                "joint_pain",
    "knee pain":                    "knee_pain",
    "knees hurt":                   "knee_pain",
    "back pain":                    "back_pain",
    "lower back pain":              "back_pain",
    "neck pain":                    "neck_pain",
    "hip pain":                     "hip_joint_pain",
    "painful walking":              "painful_walking",
    "pain when walking":            "painful_walking",
    "limb weakness":                "weakness_in_limbs",
    "arms and legs weak":           "weakness_in_limbs",
    "legs feel weak":               "weakness_in_limbs",
    "movement stiffness":           "movement_stiffness",
    "stiff joints":                 "movement_stiffness",
    "stiff muscles":                "movement_stiffness",

    # ── Swelling / physical ───────────────────────────────────
    "puffy face":                   "puffy_face_and_eyes",
    "swollen face":                 "puffy_face_and_eyes",
    "face is swollen":              "puffy_face_and_eyes",
    "swollen stomach":              "swelling_of_stomach",
    "stomach is swollen":           "swelling_of_stomach",
    "swollen joints":               "swelling_joints",
    "joints are swollen":           "swelling_joints",
    "enlarged thyroid":             "enlarged_thyroid",
    "thyroid swelling":             "enlarged_thyroid",
    "neck swelling":                "enlarged_thyroid",

    # ── Sweating / chills ─────────────────────────────────────
    "sweating a lot":               "sweating",
    "heavy sweating":               "sweating",
    "excessive sweating":           "sweating",
    "night sweats":                 "sweating",
    "chills":                       "chills",
    "feeling cold":                 "chills",
    "shivering":                    "shivering",
    "trembling":                    "shivering",
    "cold hands":                   "cold_hands_and_feets",
    "cold feet":                    "cold_hands_and_feets",
    "cold hands and feet":          "cold_hands_and_feets",

    # ── Liver / jaundice related ──────────────────────────────
    "yellow skin":                  "yellowish_skin",
    "skin is yellow":               "yellowish_skin",
    "liver failure":                "acute_liver_failure",
    "liver problem":                "acute_liver_failure",
    "fluid retention":              "fluid_overload",
    "retaining fluid":              "fluid_overload",
    "body is swollen":              "fluid_overload",

    # ── Obesity / weight ──────────────────────────────────────
    "overweight":                   "obesity",
    "obese":                        "obesity",
    "very overweight":              "obesity",

    # ── Nails / hair ──────────────────────────────────────────
    "brittle nails":                "brittle_nails",
    "nails breaking":               "brittle_nails",
    "nail dents":                   "small_dents_in_nails",
    "dents in nails":               "small_dents_in_nails",
    "inflamed nails":               "inflammatory_nails",

    # ── Menstrual ─────────────────────────────────────────────
    "irregular periods":            "abnormal_menstruation",
    "abnormal periods":             "abnormal_menstruation",
    "period problems":              "abnormal_menstruation",
    "menstrual problems":           "abnormal_menstruation",

    # ── Misc ──────────────────────────────────────────────────
    "malaise":                      "malaise",
    "feeling unwell":               "malaise",
    "generally unwell":             "malaise",
    "not feeling well":             "malaise",
    "cramps":                       "cramps",
    "muscle cramps":                "cramps",
    "stomach cramps":               "cramps",
    "toxic look":                   "toxic_look_(typhos)",
    "looks very sick":              "toxic_look_(typhos)",
    "family history":               "family_history",
    "runs in family":               "family_history",
}

# ── Load models ───────────────────────────────────────────────
model         = joblib.load('models/best_model.pkl')
symptom_list  = joblib.load('models/symptom_list.pkl')
nn_model      = tf.keras.models.load_model('models/neural_network.keras')
label_encoder = joblib.load('models/label_encoder.pkl')


# ══════════════════════════════════════════════════════════════
# INPUT HELPER
# ══════════════════════════════════════════════════════════════
def get_input_df(selected_symptoms):
    """Converts selected symptoms to binary DataFrame."""
    input_vector = [1 if sym in selected_symptoms else 0 for sym in symptom_list]
    return pd.DataFrame([input_vector], columns=symptom_list)


# ══════════════════════════════════════════════════════════════
# SYMPTOM EXTRACTION — TEXT (NLP)
# ══════════════════════════════════════════════════════════════
def extract_symptoms_from_text(user_text):
    """
    Extracts matching symptoms from natural language text.
    Uses alias dictionary, direct matching, and fuzzy matching.
    """
    if not user_text:
        return []

    user_text_clean = user_text.lower().strip()
    found = []

    # Method 0 — Alias dictionary (catches semantic gaps fuzzy can't handle)
    for phrase, symptom in SYMPTOM_ALIASES.items():
        if phrase in user_text_clean:
            if symptom in symptom_list and symptom not in found:
                found.append(symptom)

    # Method 1 — Direct keyword match (unchanged)
    for symptom in symptom_list:
        sym_underscored = symptom.lower()
        sym_spaced      = symptom.lower().replace('_', ' ')
        if sym_underscored in user_text_clean or sym_spaced in user_text_clean:
            if symptom not in found:
                found.append(symptom)

    # Method 2 — Fuzzy matching (unchanged)
    words  = user_text_clean.replace(',', ' ').replace('.', ' ').replace('-', ' ').split()
    ngrams = []
    for i in range(len(words)):
        ngrams.append(words[i])
        if i + 1 < len(words):
            ngrams.append(words[i] + ' ' + words[i+1])
        if i + 2 < len(words):
            ngrams.append(words[i] + ' ' + words[i+1] + ' ' + words[i+2])

    symptom_list_clean = [s.replace('_', ' ') for s in symptom_list]
    for ngram in ngrams:
        match = process.extractOne(
            ngram,
            symptom_list_clean,
            scorer=fuzz.ratio,
            score_cutoff=72
        )
        if match:
            matched_symptom = symptom_list[symptom_list_clean.index(match[0])]
            if matched_symptom not in found:
                found.append(matched_symptom)

    return found

# ══════════════════════════════════════════════════════════════
# SYMPTOM EXTRACTION — DOCUMENT
# ══════════════════════════════════════════════════════════════
def extract_text_from_file(file_bytes, file_name):
    """
    Extracts raw text from PDF or image file bytes.
    Returns extracted text string.
    """
    text = ""
    file_name = file_name.lower()

    if file_name.endswith('.pdf'):
        try:
            import fitz  # PyMuPDF
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in pdf_doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text + " "
            pdf_doc.close()
        except Exception as e:
            print(f"PyMuPDF failed: {e}")

        # Fallback to pdfplumber if PyMuPDF got nothing
        if not text.strip():
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + " "
            except Exception as e:
                print(f"pdfplumber failed: {e}")

    elif file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
        try:
            import pytesseract
            from PIL import Image
            if platform.system() == 'Windows':
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            image = Image.open(io.BytesIO(file_bytes))
            text  = pytesseract.image_to_string(image)
        except Exception as e:
            print(f"Tesseract OCR failed: {e}")

    return text.strip()


def extract_symptoms_from_document_direct(text):
    """
    Direct symptom extraction from document text.
    Checks exact underscore and space forms.
    """
    if not text:
        return []

    text_lower = text.lower()
    found = []

    for symptom in symptom_list:
        sym_underscored = symptom.lower()
        sym_spaced      = symptom.lower().replace('_', ' ')

        if sym_underscored in text_lower or sym_spaced in text_lower:
            if symptom not in found:
                found.append(symptom)

    return found


def extract_symptoms_from_document(file):
    """
    Main document processing function.
    Reads file, extracts text, matches symptoms.
    Uses direct matching + full NLP (alias + fuzzy) always.
    Returns (symptoms_list, preview_text).
    """
    # Read file bytes
    file_bytes = file.read()
    file_name  = file.filename if hasattr(file, 'filename') else file.name

    print("=" * 50)
    print(f"FILE NAME: {file_name}")
    print(f"FILE BYTES LENGTH: {len(file_bytes)}")

    if len(file_bytes) == 0:
        return [], "File is empty or could not be read."

    # Extract raw text from file
    text = extract_text_from_file(file_bytes, file_name)

    print("EXTRACTED TEXT PREVIEW:")
    print(text[:300] if text else "(empty)")
    print("=" * 50)

    if not text.strip():
        return [], "No readable text found in the document."

    # Method 1 — Direct matching (exact symptom names from dataset)
    found = extract_symptoms_from_document_direct(text)
    print(f"DIRECT MATCH found ({len(found)}): {found}")

    # Method 2 & 3 — Alias dictionary + fuzzy (always runs, not just fallback)
    full_found = extract_symptoms_from_text(text)
    for s in full_found:
        if s not in found:
            found.append(s)

    print(f"FINAL SYMPTOMS ({len(found)}): {found}")
    print("=" * 50)

    return found, text[:500]

# ══════════════════════════════════════════════════════════════
# PREDICTIONS
# ══════════════════════════════════════════════════════════════
def predict_top3_diseases(selected_symptoms):
    """Returns top 3 predicted diseases with confidence from Random Forest."""
    input_df      = get_input_df(selected_symptoms)
    probabilities = model.predict_proba(input_df)[0]
    classes       = model.classes_
    disease_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return [(disease, round(prob * 100, 2)) for disease, prob in disease_probs[:3]]


def predict_nn_top3(selected_symptoms):
    """Returns top 3 predicted diseases with confidence from Neural Network."""
    input_vector  = [1 if sym in selected_symptoms else 0 for sym in symptom_list]
    input_array   = np.array([input_vector])
    probabilities = nn_model.predict(input_array, verbose=0)[0]
    top_indices   = np.argsort(probabilities)[::-1][:3]
    return [
        (label_encoder.classes_[i], round(float(probabilities[i]) * 100, 2))
        for i in top_indices
    ]


def predict_all_models(selected_symptoms):
    """Returns prediction from RF and Neural Network for comparison."""
    input_df = get_input_df(selected_symptoms)

    results = {}

    # Random Forest
    rf_pred = model.predict(input_df)[0]
    rf_conf = round(model.predict_proba(input_df).max() * 100, 2)
    results['🤖 Random Forest'] = {'disease': rf_pred, 'confidence': rf_conf}

    # Neural Network
    input_array   = np.array([input_df.values[0]])
    probabilities = nn_model.predict(input_array, verbose=0)[0]
    nn_pred_idx   = np.argmax(probabilities)
    nn_pred       = label_encoder.classes_[nn_pred_idx]
    nn_conf       = round(float(probabilities[nn_pred_idx]) * 100, 2)
    results['🧠 Neural Network'] = {'disease': nn_pred, 'confidence': nn_conf}

    return results


# ══════════════════════════════════════════════════════════════
# DISEASE INFO
# ══════════════════════════════════════════════════════════════
def get_disease_info(disease_name, desc_df, prec_df):
    """Returns description and precautions for a disease."""
    desc_row    = desc_df[desc_df['Disease'].str.strip() == disease_name]
    description = desc_row['Description'].values[0] if not desc_row.empty else "No description available."
    prec_row    = prec_df[prec_df['Disease'].str.strip() == disease_name]
    precautions = []
    if not prec_row.empty:
        for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
            val = prec_row[col].values[0]
            if pd.notna(val) and str(val).strip() != '':
                precautions.append(val.strip())
    return description, precautions


# ══════════════════════════════════════════════════════════════
# SEVERITY
# ══════════════════════════════════════════════════════════════
def get_severity_score(selected_symptoms, severity_df):
    """Calculates average severity score and risk level."""
    total   = 0
    matched = 0
    for sym in selected_symptoms:
        row = severity_df[severity_df['Symptom'].str.strip() == sym]
        if not row.empty:
            total   += row['weight'].values[0]
            matched += 1
    if matched == 0:
        return 0, "Unknown"
    avg  = round(total / matched, 2)
    risk = "High" if avg >= 5 else "Medium" if avg >= 3 else "Low"
    return avg, risk


# ══════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════
def generate_pdf_report(
    selected_symptoms,
    top3,
    description,
    precautions,
    severity_score,
    risk_level
):
    """Generate professional AI healthcare assessment PDF."""

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable
    )
    from reportlab.lib.units import cm

    # =========================
    # PDF DOCUMENT SETUP
    # =========================
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    # =========================
    # COLORS
    # =========================
    PRIMARY = colors.HexColor('#1E3A5F')
    LIGHT_BG = colors.HexColor('#F7FAFC')
    BORDER = colors.HexColor('#D6E0EA')
    TEXT_GRAY = colors.HexColor('#6C757D')
    SUCCESS = colors.HexColor('#1FA67A')

    # =========================
    # STYLES
    # =========================
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=5
    )

    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=TEXT_GRAY,
        spaceAfter=14
    )

    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=16,
        textColor=colors.black
    )

    info_style = ParagraphStyle(
        'InfoStyle',
        parent=styles['BodyText'],
        fontSize=10,
        leading=18,
        textColor=colors.black
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['BodyText'],
        fontSize=8,
        leading=12,
        textColor=TEXT_GRAY
    )

    # =========================
    # TITLE SECTION
    # =========================
    story.append(
        Paragraph(
            "AI-Powered Health Assessment Report",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Advanced Symptom Analysis & Clinical Risk Prediction",
            subtitle_style
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=BORDER
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    # =========================
    # SYMPTOM SUMMARY
    # =========================
    story.append(
        Paragraph(
            "Patient Symptom Summary",
            heading_style
        )
    )

    formatted_symptoms = " • ".join(
        sym.replace("_", " ").title()
        for sym in selected_symptoms
    )

    story.append(
        Paragraph(
            formatted_symptoms,
            normal_style
        )
    )

    story.append(Spacer(1, 0.35 * cm))

    # =========================
    # RISK ASSESSMENT
    # =========================
    story.append(
        Paragraph(
            "Overall Health Risk Assessment",
            heading_style
        )
    )

    risk_text = f"""
    <b>Risk Level:</b> {risk_level}<br/>
    <b>Severity Score:</b> {severity_score} / 7<br/>
    <b>Clinical Observation:</b> The entered symptoms may indicate conditions requiring medical evaluation and professional consultation.
    """

    story.append(
        Paragraph(
            risk_text,
            info_style
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    # =========================
    # TOP PREDICTIONS TABLE
    # =========================
    story.append(
        Paragraph(
            "Top Predicted Conditions",
            heading_style
        )
    )

    prediction_data = [
        ['Priority', 'Predicted Condition', 'AI Confidence']
    ]

    priorities = ['1st', '2nd', '3rd']

    for rank, (disease, confidence) in zip(priorities, top3):
        prediction_data.append([
            rank,
            disease,
            f"{confidence}%"
        ])

    prediction_table = Table(
        prediction_data,
        colWidths=[3 * cm, 9 * cm, 4 * cm]
    )

    prediction_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),

        ('BACKGROUND', (0, 1), (-1, -1), LIGHT_BG),

        ('GRID', (0, 0), (-1, -1), 0.6, BORDER),

        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),

        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))

    story.append(prediction_table)

    story.append(Spacer(1, 0.5 * cm))

    # =========================
    # CONDITION ANALYSIS
    # =========================
    top_condition = top3[0][0]
    top_confidence = top3[0][1]

    story.append(
        Paragraph(
            "Detailed Condition Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"<b>{top_condition}</b>",
            styles['Heading3']
        )
    )

    story.append(Spacer(1, 0.1 * cm))

    story.append(
        Paragraph(
            f"<b>Estimated Risk:</b> {top_confidence}% AI Confidence",
            info_style
        )
    )

    story.append(Spacer(1, 0.2 * cm))

    story.append(
        Paragraph(
            "<b>Condition Overview</b>",
            normal_style
        )
    )

    story.append(
        Paragraph(
            description,
            normal_style
        )
    )

    story.append(Spacer(1, 0.4 * cm))

    # =========================
    # PRECAUTIONS
    # =========================
    story.append(
        Paragraph(
            "Recommended Immediate Actions",
            heading_style
        )
    )

    for i, precaution in enumerate(precautions, 1):
        story.append(
            Paragraph(
                f"{i}. {precaution.capitalize()}",
                normal_style
            )
        )

    story.append(Spacer(1, 0.5 * cm))

    # =========================
    # HEALTH RECOMMENDATIONS
    # =========================
    story.append(
        Paragraph(
            "General Health Recommendations",
            heading_style
        )
    )

    recommendations = [
        "Maintain proper hydration and rest",
        "Monitor symptoms regularly",
        "Avoid unnecessary physical stress",
        "Seek professional medical consultation if symptoms worsen",
        "Follow a balanced and healthy lifestyle"
    ]

    for rec in recommendations:
        story.append(
            Paragraph(
                f"• {rec}",
                normal_style
            )
        )

    story.append(Spacer(1, 0.6 * cm))

    # =========================
    # DISCLAIMER
    # =========================
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=BORDER
        )
    )

    story.append(Spacer(1, 0.25 * cm))

    disclaimer = """
    <b>Important Disclaimer:</b>
    This AI-generated assessment is intended for educational and research purposes only.
    It is not a substitute for professional medical diagnosis, treatment, or emergency care.
    Please consult a licensed healthcare professional for proper medical advice and treatment.
    """

    story.append(
        Paragraph(
            disclaimer,
            disclaimer_style
        )
    )

    # =========================
    # BUILD PDF
    # =========================
    doc.build(story)

    buffer.seek(0)

    return buffer