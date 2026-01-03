from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import time
import requests
import os

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

@app.route('/')
def home():
    """Serve the HTML interface"""
    return send_file('index.html')

conversation_history = []
MAX_HISTORY_LENGTH = 20

# ====== OPENROUTER API CONFIGURATION ======
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_MODEL = "anthropic/claude-3-haiku"
OPENROUTER_API_URL = "openrouter.ai"

USE_REAL_AI = True

def construct_openrouter_headers():
    """Construct headers for OpenRouter API"""
    if not OPENROUTER_API_KEY:
        return None
    
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "AI Learning Assistant"
    }

def get_openrouter_response(question):
    """Get real AI response from OpenRouter API"""
    if not OPENROUTER_API_KEY:
        return "❌ API Key Error: No OpenRouter API key configured. Please set the OPENROUTER_API_KEY environment variable."
    
    print(f"[AI] Getting response for: {question[:50]}...")
    
    headers = construct_openrouter_headers()
    if not headers:
        return "❌ API configuration error. Please check your API key."
    
    system_prompt = """You are a friendly AI tutor for students named "LearnBot". 

Your teaching principles:
1. EXPLAIN CLEARLY: Break down complex topics into simple, digestible parts
2. PROVIDE EXAMPLES: Always include practical, real-world examples
3. ENCOURAGE LEARNING: Use positive reinforcement and encouragement
4. ADAPT TO LEVEL: Assess the student's level from their question
5. USE MARKDOWN: Format responses with headings, lists, and code blocks

Response format:
- Start with a friendly greeting
- Explain the concept step-by-step
- Include at least one practical example
- End with an encouraging note or follow-up question
- Use **bold** for key terms and `code blocks` for code examples

Remember: The student is here to learn. Be patient, supportive, and educational!"""
    
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "max_tokens": 2000,
        "temperature": 0.7,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            print(f"[AI] Sending request to OpenRouter API (Attempt {attempt + 1}/{max_retries})...")
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=45
            )
            
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"[AI] Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "⚠️ Rate limit exceeded. Please wait a moment and try again."
            
            if response.status_code == 401:
                return "❌ Invalid API key. Please check your OpenRouter API key."
            elif response.status_code == 402:
                return "❌ Payment required. You may need to add credits to your OpenRouter account."
            elif response.status_code == 400:
                error_detail = response.json().get('error', {}).get('message', 'Bad request')
                return f"⚠️ Bad request: {error_detail}"
            
            response.raise_for_status()
            result = response.json()
            
            ai_response = result["choices"][0]["message"]["content"]
            print(f"[AI] Received response ({len(ai_response)} characters)")
            
            if "usage" in result:
                prompt_tokens = result["usage"].get("prompt_tokens", 0)
                completion_tokens = result["usage"].get("completion_tokens", 0)
                print(f"[AI] Token usage: {prompt_tokens} + {completion_tokens} = {prompt_tokens + completion_tokens}")
            
            return ai_response
            
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                status_code = response.status_code if 'response' in locals() else 'Unknown'
                return f"⚠️ HTTP Error {status_code}: {str(e)}"
            continue
            
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                return "🌐 Connection error. Please check your internet connection."
            time.sleep(base_delay * (2 ** attempt))
            continue
            
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return "⏰ Request timed out. Please try again."
            time.sleep(base_delay * (2 ** attempt))
            continue
            
        except KeyError as e:
            error_msg = f"⚠️ API Response Error: Missing key {str(e)}"
            print(f"[ERROR] {error_msg}")
            if 'result' in locals():
                print(f"[ERROR] Full response: {result}")
            return f"{error_msg}\n\nThe API returned an unexpected response."
            
        except Exception as e:
            error_msg = f"⚠️ Unexpected Error: {type(e).__name__}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg
    
    return "⚠️ Failed to get response after multiple attempts. Please try again later."

def generate_mock_response(question):
    """Fallback mock responses if API fails"""
    import random
    mock_responses = [
        f"""**LearnBot Mock Response** 🤖

You asked: "{question}"

I'd love to help you learn about this! In real AI mode, I would:

1. Break down the concept into simple parts
2. Provide clear examples and analogies
3. Offer practical applications
4. Suggest next steps for deeper learning

*To enable real AI responses:*
1. Get a free OpenRouter API key from: openrouter.ai
2. Set it as environment variable: `set OPENROUTER_API_KEY=your_key_here` (Windows)
3. Restart this server

What specific aspect interests you the most?""",
        
        f"""**AI Tutor (Mock Mode)** 📚

Question: {question}

Great question! This is exactly the type of topic I enjoy explaining.

Available models on OpenRouter:
- Claude 3 Haiku (fast & capable) - Currently selected
- Gemini Flash (Google's model)
- Llama 3.2 (Meta's open model)
- Many other options!

*OpenRouter gives you access to multiple AI models with one API key.*

Ready to learn? Enable real AI mode and ask away! 🚀"""
    ]
    
    return random.choice(mock_responses)

@app.route('/health', methods=['GET'])
def health_check():
    """Check server and API status"""
    api_configured = bool(OPENROUTER_API_KEY)
    key_preview = f"{OPENROUTER_API_KEY[:10]}..." if OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 10 else "Not set"
    
    return jsonify({
        "status": "healthy",
        "backend": "Flask Server",
        "version": "5.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ai_api_key_status": key_preview
