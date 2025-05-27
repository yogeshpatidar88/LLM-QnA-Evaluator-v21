from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
from ass import AdvancedQAEvaluator

# Load environment variables from .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed - using system environment variables only")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize evaluator
evaluator = None

try:
    # Remove hardcoded API key - use environment variable
    evaluator = AdvancedQAEvaluator()  # Will use GEMINI_API_KEY from environment
    print("✅ Advanced evaluator initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing evaluator: {e}")
    print("Make sure GEMINI_API_KEY environment variable is set")

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to ask a question and get evaluation"""
    if not evaluator:
        return jsonify({"error": "Evaluator not initialized - check API key"}), 500
    
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    # Get answer
    answer = evaluator.get_answer(question)
    
    # Evaluate answer with advanced metrics
    evaluation = None
    if not answer.startswith("Error getting answer:"):
        evaluation = evaluator.evaluate_answer_advanced(question, answer)
    
    return jsonify({
        "question": question,
        "answer": answer,
        "evaluation": evaluation,
        "timestamp": str(datetime.now())
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "evaluator_ready": evaluator is not None})

if __name__ == '__main__':
    # For production deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV') != 'production')