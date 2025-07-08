from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# Initialize Rouge scorer
rouge = Rouge()

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate texts"""
    try:
        smoothing_function = SmoothingFunction().method1
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing_function)
        return bleu_score
    except Exception as e:
        return 0.0

def calculate_rouge(reference, candidate):
    """Calculate ROUGE score between reference and candidate texts"""
    try:
        scores = rouge.get_scores(candidate, reference)[0]
        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }
    except Exception as e:
        return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}

def generate_feedback(student_answer, model_answer, bleu_score, rouge_scores):
    """Generate feedback using Google Gemini"""
    try:
        prompt = f"""
        Provide brief, concise feedback for a student's answer. Keep it SHORT and direct.
        
        Student's Answer: {student_answer}
        Model Answer: {model_answer}
        Scores: BLEU={bleu_score:.3f}, ROUGE-1={rouge_scores['rouge-1']:.3f}
        
        Give feedback in this EXACT format (maximum 3-4 sentences total):
        
        **Correct:** [What student got right - 1 sentence]
        **Missing:** [Key points student missed - 1 sentence]  
        **Grade:** [A/B/C/D/F with brief reason - 1 sentence]
        **Tip:** [One actionable improvement - 1 sentence]
        
        Be concise and direct. No lengthy explanations.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating feedback: {str(e)}"

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Grading System API",
        "version": "1.0.0",
        "endpoints": {
            "POST /grade": "Grade student answer against model answer",
            "GET /health": "Health check endpoint"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "API is running"})

@app.route('/grade', methods=['POST'])
def grade_answer():
    """Grade student answer against model answer"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        student_answer = data.get('student_answer', '').strip()
        model_answer = data.get('model_answer', '').strip()
        
        if not student_answer or not model_answer:
            return jsonify({"error": "Both student_answer and model_answer are required"}), 400
        
        # Calculate scores
        bleu_score = calculate_bleu(model_answer, student_answer)
        rouge_scores = calculate_rouge(model_answer, student_answer)
        
        # Generate feedback
        feedback = generate_feedback(student_answer, model_answer, bleu_score, rouge_scores)
        
        # Prepare response
        response = {
            "student_answer": student_answer,
            "model_answer": model_answer,
            "scores": {
                "bleu": round(bleu_score, 4),
                "rouge-1": round(rouge_scores['rouge-1'], 4),
                "rouge-2": round(rouge_scores['rouge-2'], 4),
                "rouge-l": round(rouge_scores['rouge-l'], 4)
            },
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    # Check if required environment variables are set
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not found in environment variables")

    app.run(debug=True, host='0.0.0.0', port=5000)
