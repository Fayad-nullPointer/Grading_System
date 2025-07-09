from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import google.generativeai as genai
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
from datetime import datetime
from bert_score import score

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

# Initialize BERTScore model once at startup (faster)
bert_scorer = None
try:
    from bert_score import BERTScorer
    # Use a smaller, faster model
    bert_scorer = BERTScorer(model_type="distilbert-base-uncased", num_layers=5)
    print("BERTScore model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load BERTScore model: {e}")
    bert_scorer = None

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

def calculate_bert_score(reference, candidate):
    """Calculate BERTScore between reference and candidate texts - Fast version"""
    try:
        if bert_scorer is None:
            return 0.0
        
        # Use pre-loaded scorer (much faster)
        P, R, F1 = bert_scorer.score([candidate], [reference])
        return float(F1[0])
    except Exception as e:
        print(f"BERTScore error: {e}")
        return 0.0

def generate_feedback(student_answer, model_answer, bleu_score, rouge_scores, bert_score):
    """Generate feedback using Google Gemini"""
    try:
        prompt = f"""
        You are an experienced teacher. Give brief and clear feedback for the student's answer in plain English.
        
        Student's Answer: {student_answer}
        Model Answer: {model_answer}
        Evaluation Scores: BLEU={bleu_score:.3f}, ROUGE-1={rouge_scores['rouge-1']:.3f}, BERTScore={bert_score:.3f}
        
        Write your evaluation naturally and simply, without any special formatting or symbols:
        
        First, mention what the student did correctly.
        Second, mention the important points they missed.
        Third, give a grade from A to F with a brief reason.
        Fourth, give one helpful tip for improvement.
        
        Do not use any symbols or formatting marks, just plain clear text.
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
        use_bert = data.get('use_bert', True)  # Option to disable BERTScore for speed
        
        if not student_answer or not model_answer:
            return jsonify({"error": "Both student_answer and model_answer are required"}), 400
        
        # Calculate scores
        bleu_score = calculate_bleu(model_answer, student_answer)
        rouge_scores = calculate_rouge(model_answer, student_answer)
        
        # Only calculate BERTScore if requested and model is available
        bert_score = 0.0
        if use_bert and bert_scorer is not None:
            bert_score = calculate_bert_score(model_answer, student_answer)
        
        # Generate feedback
        feedback = generate_feedback(student_answer, model_answer, bleu_score, rouge_scores, bert_score)
        
        # Prepare response
        response = {
            "student_answer": student_answer,
            "model_answer": model_answer,
            "scores": {
                "bleu": round(bleu_score, 4),
                "rouge-1": round(rouge_scores['rouge-1'], 4),
                "rouge-2": round(rouge_scores['rouge-2'], 4),
                "rouge-l": round(rouge_scores['rouge-l'], 4),
                "bert_score": round(bert_score, 4)
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
