import google.generativeai as genai
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, List
import nltk
from rouge_score import rouge_scorer
from sacrebleu import BLEU
import re
import os  # Add this import

# Download only what we need (small downloads)
def setup_nltk():
    """Download only required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        print("✅ NLTK punkt tokenizer already available")
    except LookupError:
        print("📥 Downloading NLTK punkt tokenizer (~13 MB)...")
        nltk.download('punkt', quiet=True)
        print("✅ Download complete!")

# Call setup function
setup_nltk()

class AdvancedQAEvaluator:
    def __init__(self, api_key: str = None):
        # Get API key from environment variable if not provided
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("API key not found. Please set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=api_key)
        
        # Initialize ROUGE scorer (no additional downloads needed)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BLEU scorer (no additional downloads needed)
        self.bleu_scorer = BLEU()
        
        # Try different model names in order
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'gemini-pro',
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro'
        ]
        
        self.model = None
        for model_name in model_names:
            try:
                print(f"🔄 Trying model: {model_name}")
                self.model = genai.GenerativeModel(model_name)
                test_response = self.model.generate_content("Hello")
                print(f"✅ Successfully loaded LLM Judge model: {model_name}")
                break
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        if self.model is None:
            raise Exception("Could not load any LLM Judge model. Check available models.")
    
    def get_answer(self, question: str) -> str:
        """Get answer from Gemini API"""
        try:
            response = self.model.generate_content(question)
            return response.text
        except Exception as e:
            return f"Error getting answer: {e}"
    
    def generate_reference_answer(self, question: str) -> str:
        """Generate a high-quality reference answer for comparison using LLM-as-a-Judge"""
        reference_prompt = f"""
        As an expert judge, provide the GOLD STANDARD answer to this question.
        This will be used as a reference for evaluation purposes.
        
        Requirements for the reference answer:
        - Maximum comprehensiveness and detail
        - Perfect structure and clarity
        - Complete coverage of all aspects
        - Actionable and specific guidance
        - Industry best practices included
        
        Question: {question}
        
        Provide the most comprehensive answer possible that covers:
        - Specific details and concrete examples
        - Step-by-step actionable guidance
        - Proper structure and organization
        - Relevant context and explanations
        - Potential pitfalls and solutions
        """
        
        try:
            response = self.model.generate_content(reference_prompt)
            return response.text
        except Exception as e:
            return "Unable to generate reference answer"
    
    def calculate_rouge_scores(self, answer: str, reference: str) -> Dict:
        """Calculate ROUGE scores between answer and reference"""
        try:
            scores = self.rouge_scorer.score(reference, answer)
            return {
                'rouge1': {
                    'precision': scores['rouge1'].precision,
                    'recall': scores['rouge1'].recall,
                    'fmeasure': scores['rouge1'].fmeasure
                },
                'rouge2': {
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'fmeasure': scores['rouge2'].fmeasure
                },
                'rougeL': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'fmeasure': scores['rougeL'].fmeasure
                }
            }
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            return {}
    
    def calculate_bleu_score(self, answer: str, reference: str) -> float:
        """Calculate BLEU score between answer and reference"""
        try:
            # Simple word tokenization (fallback if NLTK punkt fails)
            try:
                answer_tokens = nltk.word_tokenize(answer.lower())
                reference_tokens = nltk.word_tokenize(reference.lower())
            except:
                # Fallback to simple split if NLTK fails
                answer_tokens = answer.lower().split()
                reference_tokens = reference.lower().split()
            
            # BLEU expects list of references
            bleu_score = self.bleu_scorer.sentence_score(
                ' '.join(answer_tokens), 
                [' '.join(reference_tokens)]
            )
            return bleu_score.score / 100.0  # Convert to 0-1 scale
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return 0.0
    
    def evaluate_answer_advanced(self, question: str, answer: str) -> Dict:
        """Comprehensive evaluation using LLM-as-a-Judge + ROUGE + BLEU"""
        if answer.startswith("Error getting answer:"):
            return {
                "quality": "ERROR",
                "score": 0,
                "reasoning": "Cannot evaluate - original answer failed",
                "missing_elements": [],
                "strengths": [],
                "rouge_scores": {},
                "bleu_score": 0.0,
                "reference_answer": "",
                "llm_judge_verdict": "ERROR",
                "evaluation_method": "LLM-as-a-Judge + NLP Metrics"
            }
        
        print("🔄 Generating reference answer using LLM-as-a-Judge...")
        reference_answer = self.generate_reference_answer(question)
        
        print("🔄 Calculating ROUGE scores...")
        rouge_scores = self.calculate_rouge_scores(answer, reference_answer)
        
        print("🔄 Calculating BLEU score...")
        bleu_score = self.calculate_bleu_score(answer, reference_answer)
        
        print("🔄 Running LLM-as-a-Judge evaluation...")
        llm_judge_eval = self.llm_as_judge_evaluation(question, answer, rouge_scores, bleu_score)
        
        return {
            **llm_judge_eval,
            "rouge_scores": rouge_scores,
            "bleu_score": bleu_score,
            "reference_answer": reference_answer,
            "metrics_summary": self.create_metrics_summary(rouge_scores, bleu_score),
            "evaluation_method": "LLM-as-a-Judge + NLP Metrics",
            "judge_model": "Gemini (LLM-as-a-Judge)"
        }
    
    def llm_as_judge_evaluation(self, question: str, answer: str, rouge_scores: Dict, bleu_score: float) -> Dict:
        """LLM-as-a-Judge evaluation with enhanced prompt engineering"""
        
        metrics_context = ""
        if rouge_scores:
            rouge1_f = rouge_scores.get('rouge1', {}).get('fmeasure', 0)
            rouge2_f = rouge_scores.get('rouge2', {}).get('fmeasure', 0)
            rougeL_f = rouge_scores.get('rougeL', {}).get('fmeasure', 0)
            
            metrics_context = f"""
            
            📊 OBJECTIVE SIMILARITY METRICS (for context):
            - ROUGE-1 F-score: {rouge1_f:.3f} (word overlap with reference)
            - ROUGE-2 F-score: {rouge2_f:.3f} (bigram overlap with reference)
            - ROUGE-L F-score: {rougeL_f:.3f} (longest common subsequence)
            - BLEU score: {bleu_score:.3f} (n-gram precision)
            
            Note: Higher scores indicate better similarity to comprehensive reference answers.
            """
        
        evaluation_prompt = f"""
        🏛️ LLM-AS-A-JUDGE EVALUATION TASK
        
        You are serving as an expert judge to evaluate the quality of a Q&A response.
        Your role is to provide objective, consistent, and detailed assessment.
        
        📋 EVALUATION CRITERIA:
        
        ✅ GOOD Answer Characteristics:
        - Specific and detailed information
        - Actionable insights and clear steps
        - Directly addresses all aspects of the question
        - Well-structured and organized
        - Comprehensive coverage of the topic
        - Includes examples or practical guidance
        
        ❌ BAD Answer Characteristics:
        - Vague, generic, or superficial responses
        - Lacks actionable steps or specific guidance
        - Doesn't fully address the question asked
        - Unclear, confusing, or poorly structured
        - Incomplete coverage of important aspects
        - Too brief or missing critical information
        
        📝 QUESTION TO EVALUATE:
        {question}
        
        📝 ANSWER TO JUDGE:
        {answer}
        {metrics_context}
        
        🎯 JUDGE'S VERDICT REQUIRED:
        
        As an LLM Judge, provide your evaluation in this EXACT JSON format:
        {{
            "llm_judge_verdict": "GOOD" or "BAD",
            "quality": "GOOD" or "BAD",
            "score": <1-10 overall quality score>,
            "reasoning": "Detailed explanation of your judgment, considering both content quality and objective metrics",
            "missing_elements": ["specific elements that would improve the answer"],
            "strengths": ["specific positive aspects of the answer"],
            "content_depth": <1-10 score>,
            "actionability": <1-10 score>,
            "clarity": <1-10 score>,
            "comprehensiveness": <1-10 score>,
            "judge_confidence": <1-10 confidence in this evaluation>
        }}
        
        Remember: You are acting as an expert judge. Be thorough, fair, and consistent.
        """
        
        try:
            response = self.model.generate_content(evaluation_prompt)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            result = json.loads(response_text)
            
            # Ensure llm_judge_verdict is present
            if 'llm_judge_verdict' not in result:
                result['llm_judge_verdict'] = result.get('quality', 'UNKNOWN')
            
            return result
        except Exception as e:
            return {
                "llm_judge_verdict": "ERROR",
                "quality": "ERROR",
                "score": 0,
                "reasoning": f"LLM Judge evaluation error: {e}",
                "missing_elements": [],
                "strengths": [],
                "content_depth": 0,
                "actionability": 0,
                "clarity": 0,
                "comprehensiveness": 0,
                "judge_confidence": 0
            }
    
    def create_metrics_summary(self, rouge_scores: Dict, bleu_score: float) -> Dict:
        """Create a summary of all metrics"""
        if not rouge_scores:
            return {"overall_similarity": 0.0, "interpretation": "No metrics available"}
        
        # Calculate weighted average of metrics
        rouge1_f = rouge_scores.get('rouge1', {}).get('fmeasure', 0)
        rouge2_f = rouge_scores.get('rouge2', {}).get('fmeasure', 0)
        rougeL_f = rouge_scores.get('rougeL', {}).get('fmeasure', 0)
        
        # Weighted combination
        overall_similarity = (rouge1_f * 0.3 + rouge2_f * 0.2 + rougeL_f * 0.3 + bleu_score * 0.2)
        
        # Interpretation
        if overall_similarity >= 0.7:
            interpretation = "Excellent similarity to comprehensive reference"
        elif overall_similarity >= 0.5:
            interpretation = "Good similarity, covers main points well"
        elif overall_similarity >= 0.3:
            interpretation = "Moderate similarity, missing some details"
        elif overall_similarity >= 0.1:
            interpretation = "Low similarity, significantly different approach"
        else:
            interpretation = "Very low similarity, may be off-topic or very brief"
        
        return {
            "overall_similarity": overall_similarity,
            "interpretation": interpretation,
            "rouge1_fmeasure": rouge1_f,
            "rouge2_fmeasure": rouge2_f,
            "rougeL_fmeasure": rougeL_f,
            "bleu_score": bleu_score
        }
    
    def display_evaluation(self, evaluation: Dict):
        """Display comprehensive evaluation results with LLM-as-a-Judge highlighting"""
        print("\n" + "="*80)
        print("🏛️ LLM-AS-A-JUDGE EVALUATION RESULTS")
        print("="*80)
        
        # LLM Judge Verdict
        llm_verdict = evaluation.get('llm_judge_verdict', 'UNKNOWN')
        judge_confidence = evaluation.get('judge_confidence', 0)
        
        print(f"⚖️  LLM JUDGE VERDICT: {llm_verdict}")
        print(f"🎯 Judge Confidence: {judge_confidence}/10")
        print(f"🤖 Judge Model: {evaluation.get('judge_model', 'Gemini')}")
        print(f"📊 Evaluation Method: {evaluation.get('evaluation_method', 'LLM-as-a-Judge + NLP Metrics')}")
        
        quality = evaluation.get('quality', 'UNKNOWN')
        score = evaluation.get('score', 0)
        
        # Quality and Score
        if quality == "GOOD":
            print(f"\n✅ FINAL QUALITY: {quality} (Overall Score: {score}/10)")
        elif quality == "BAD":
            print(f"\n❌ FINAL QUALITY: {quality} (Overall Score: {score}/10)")
        else:
            print(f"\n⚠️  FINAL QUALITY: {quality} (Overall Score: {score}/10)")
        
        # Detailed scores from LLM Judge
        content_depth = evaluation.get('content_depth', 0)
        actionability = evaluation.get('actionability', 0)
        clarity = evaluation.get('clarity', 0)
        comprehensiveness = evaluation.get('comprehensiveness', 0)
        
        print(f"\n📋 LLM JUDGE DETAILED SCORES:")
        print(f"   Content Depth: {content_depth}/10")
        print(f"   Actionability: {actionability}/10")
        print(f"   Clarity: {clarity}/10")
        print(f"   Comprehensiveness: {comprehensiveness}/10")
        
        # Similarity Metrics
        metrics_summary = evaluation.get('metrics_summary', {})
        if metrics_summary:
            print(f"\n📈 OBJECTIVE NLP METRICS:")
            print(f"   Overall Similarity: {metrics_summary.get('overall_similarity', 0):.3f}")
            print(f"   ROUGE-1 F-measure: {metrics_summary.get('rouge1_fmeasure', 0):.3f}")
            print(f"   ROUGE-2 F-measure: {metrics_summary.get('rouge2_fmeasure', 0):.3f}")
            print(f"   ROUGE-L F-measure: {metrics_summary.get('rougeL_fmeasure', 0):.3f}")
            print(f"   BLEU Score: {metrics_summary.get('bleu_score', 0):.3f}")
            print(f"   📝 {metrics_summary.get('interpretation', 'No interpretation available')}")
        
        # LLM Judge Reasoning
        print(f"\n🧠 LLM JUDGE REASONING:")
        print(f"   {evaluation.get('reasoning', 'No reasoning provided')}")
        
        # Strengths
        if evaluation.get('strengths'):
            print(f"\n✨ IDENTIFIED STRENGTHS:")
            for strength in evaluation['strengths']:
                print(f"   • {strength}")
        
        # Missing elements
        if evaluation.get('missing_elements'):
            print(f"\n🔧 IMPROVEMENT SUGGESTIONS:")
            for element in evaluation['missing_elements']:
                print(f"   • {element}")
        
        print("="*80)
        print("💡 This evaluation combines LLM-as-a-Judge with objective NLP metrics")
        print("="*80)

# Keep the original class for backward compatibility
class TerminalQAEvaluator(AdvancedQAEvaluator):
    """Backward compatible class"""
    def evaluate_answer(self, question: str, answer: str) -> Dict:
        """Use the advanced evaluation method"""
        return self.evaluate_answer_advanced(question, answer)

def list_available_models(api_key: str):
    """Helper function to see available models"""
    genai.configure(api_key=api_key)
    print("🔍 Available LLM Judge models:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"   • {model.name}")

def main():
    print("🏛️ LLM-AS-A-JUDGE Q&A EVALUATOR")
    print("Advanced evaluation using LLM-as-a-Judge + ROUGE + BLEU metrics")
    print("Type 'quit' to exit\n")
    
    # Remove the hardcoded API key - now it will use environment variable
    try:
        evaluator = AdvancedQAEvaluator()  # No API key needed - will use env var
        print("✅ LLM-as-a-Judge evaluator ready!")
    except Exception as e:
        print(f"❌ Error setting up LLM Judge: {e}")
        print("\nMake sure to set GEMINI_API_KEY environment variable")
        return
    
    while True:
        print("\n" + "-"*50)
        question = input("🤔 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question:
            print("❌ Please enter a question!")
            continue
        
        print("\n🔄 Getting answer...")
        answer = evaluator.get_answer(question)
        
        print("\n📝 Answer:")
        print("-" * 30)
        print(answer)
        print("-" * 30)
        
        # Only evaluate if we got a real answer
        if not answer.startswith("Error getting answer:"):
            print("\n🔄 Running LLM-as-a-Judge evaluation...")
            evaluation = evaluator.evaluate_answer_advanced(question, answer)
            evaluator.display_evaluation(evaluation)
        else:
            print("❌ Skipping evaluation due to answer error")
        
        # Ask if user wants to continue
        continue_choice = input("\nPress Enter to ask another question (or 'q' to quit): ").strip()
        if continue_choice.lower() in ['q', 'quit']:
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()