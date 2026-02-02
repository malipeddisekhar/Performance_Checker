from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os
import logging
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# --- Load Models Safely ---
try:
    regressor = joblib.load(os.path.join(MODEL_DIR, "RandomForestRegressor.pkl"))
    classifier = joblib.load(os.path.join(MODEL_DIR, "RandomForestClassifier.pkl"))
    logging.info("Models loaded successfully.")
except FileNotFoundError as e:
    logging.error("Model files not found: %s", e)
    raise

# --- Expected Feature Names ---
REG_FEATURES = list(getattr(regressor, "feature_names_in_", []))
CLF_FEATURES = list(getattr(classifier, "feature_names_in_", []))

logging.info(f"Regressor features: {REG_FEATURES}")
logging.info(f"Classifier features: {CLF_FEATURES}")

# --- Risk Labels ---
risk_labels = [
    "Critical: Your profile needs urgent attention. Immediate action is required.",
    "Warning: You are at high risk. Significant improvement is needed.",
    "You're at moderate risk. Let’s work on strengthening your profile.",
    "Good progress, but there's room for improvement in some areas.",
    "Excellent! Your activity score reflects outstanding academic and extracurricular balance.",
]


# --- Helper Functions ---
def get_suggestions(data, score):
    """Generate personalized suggestions based on input data and score."""
    suggestions = []
    
    attendance = float(data.get("attendance", 0))
    cgpa = float(data.get("cgpa", 0))
    library_usage = float(data.get("library_usage", 0))
    internships = float(data.get("internships", 0))
    certificates = float(data.get("certificates", 0))
    project_involvement = float(data.get("project_involvement", 0))
    extra_curricular = float(data.get("extra_curricular", 0))
    
    if attendance < 80:
        suggestions.append("Improve your attendance to at least 85% for better academic performance.")
    if cgpa < 7.0:
        suggestions.append("Focus on improving your CGPA through consistent study habits.")
    if library_usage < 20:
        suggestions.append("Increase library usage to at least 20 hours/month for better research skills.")
    if internships < 1:
        suggestions.append("Consider applying for internships to gain practical experience.")
    if certificates < 2:
        suggestions.append("Earn more certifications to enhance your profile.")
    if project_involvement < 1:
        suggestions.append("Get involved in projects to develop practical skills.")
    if extra_curricular < 5:
        suggestions.append("Participate in more extracurricular activities for a balanced profile.")
    
    if not suggestions:
        suggestions.append("Great job! Keep maintaining your excellent academic profile.")
    
    return suggestions


def get_risk_level_from_score(score):
    """Determine risk level based on activity score."""
    if score >= 8:
        return "Low Risk"
    elif score >= 6:
        return "Moderate Risk"
    elif score >= 4:
        return "High Risk"
    else:
        return "Critical Risk"


def get_message_from_score(score):
    """Get encouraging message based on score."""
    if score >= 8:
        return "Excellent! You're on track for outstanding performance."
    elif score >= 6:
        return "Good progress! Keep working to improve further."
    elif score >= 4:
        return "You need to focus more on your academics and activities."
    else:
        return "Urgent attention needed! Please seek guidance from your advisor."


def process_inputs(data):
    """Process raw inputs and prepare for model prediction."""
    # Extract raw values (frontend sends unscaled data)
    raw_fields = {
        "attendance": float(data.get("attendance", 75)),
        "cgpa": float(data.get("cgpa", 2.5)),
        "certificates": float(data.get("certificates", 0)),
        "internships": float(data.get("internships", 0)),
        "extra_curricular": float(data.get("extra_curricular", 5)),
        "library_usage": float(data.get("library_usage", 10)),
        "project_involvement": float(data.get("project_involvement", 0)),
        "gpa_sem1": float(data.get("gpa_sem1", 2.5)),
        "gpa_sem2": float(data.get("gpa_sem2", 2.5)),
    }
    
    # Scale fields for model input
    scaled_fields = {
        "attendance": (raw_fields["attendance"] / 100) * 10,
        "cgpa": raw_fields["cgpa"],  # Already on 10-point scale
        "certificates": raw_fields["certificates"],
        "internships": raw_fields["internships"],
        "extra_curricular": raw_fields["extra_curricular"],
        "library_usage": (raw_fields["library_usage"] / 60.0) * 10,
        "project_involvement": raw_fields["project_involvement"],
        "gpa_sem1": raw_fields["gpa_sem1"],  # Already on 10-point scale
        "gpa_sem2": raw_fields["gpa_sem2"],  # Already on 10-point scale
    }
    
    # Calculate activity score using weights
    weights = {
        "attendance": 0.15,
        "cgpa": 0.2,
        "certificates": 0.1,
        "internships": 0.15,
        "extra_curricular": 0.05,
        "library_usage": 0.1,
        "project_involvement": 0.15,
        "gpa_sem1": 0.05,
        "gpa_sem2": 0.05,
    }
    activity_score = round(sum(scaled_fields[k] * weights[k] for k in weights), 2)
    
    # Prepare regressor input
    if REG_FEATURES:
        reg_input = np.array([[scaled_fields.get(f, 0) for f in REG_FEATURES]])
    else:
        reg_input = np.array([[scaled_fields[k] for k in scaled_fields]])
    
    # Prepare classifier input
    clf_input_data = {**scaled_fields, "activity_score": activity_score}
    if CLF_FEATURES:
        clf_input = np.array([[clf_input_data.get(f, 0) for f in CLF_FEATURES]])
    else:
        clf_input = np.array([[clf_input_data[k] for k in clf_input_data]])
    
    return raw_fields, scaled_fields, reg_input, clf_input, activity_score


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictor")
def predictor_page():
    return render_template("predictor.html")


@app.route("/classifier")
def classifier_page():
    return render_template("classifier.html")


@app.route("/predict_score", methods=["POST"])
def predict_score():
    try:
        data = request.get_json()
        logging.info(f"Received data for score prediction: {data}")
        
        raw_fields, scaled_fields, reg_input, _, activity_score = process_inputs(data)
        
        # Predict using regressor
        predicted_score = regressor.predict(reg_input)[0]
        predicted_score = round(float(predicted_score), 2)
        
        # Cap score between 0 and 10
        predicted_score = max(0, min(10, predicted_score))
        
        # Get suggestions and risk level
        suggestions = get_suggestions(raw_fields, predicted_score)
        risk_level = get_risk_level_from_score(predicted_score)
        message = get_message_from_score(predicted_score)
        
        response = {
            "score": predicted_score,
            "activity_score": activity_score,
            "message": message,
            "risk_level": risk_level,
            "suggestions": suggestions
        }
        
        logging.info(f"Score prediction response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in /predict_score: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    try:
        data = request.get_json()
        logging.info(f"Received data for risk prediction: {data}")
        
        raw_fields, scaled_fields, _, clf_input, activity_score = process_inputs(data)
        
        # Predict using classifier
        risk_idx = int(classifier.predict(clf_input)[0])
        risk_label = risk_labels[risk_idx] if 0 <= risk_idx < len(risk_labels) else "Unknown"
        
        # Get suggestions
        suggestions = get_suggestions(raw_fields, activity_score)
        
        response = {
            "risk_level": risk_label,
            "risk_index": risk_idx,
            "activity_score": activity_score,
            "suggestions": suggestions
        }
        
        logging.info(f"Risk prediction response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error in /predict_risk: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
