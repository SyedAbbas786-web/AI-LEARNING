from flask import Flask, request, jsonify, send_file  # Add send_file here
from flask_cors import CORS
import time
import requests
import os

app = Flask(__name__, static_folder='.', static_url_path='')  # Add this to serve static files
CORS(app)

@app.route('/')
def home():
    """Serve the HTML interface"""
    return send_file('index.html')

conversation_history = []
MAX_HISTORY_LENGTH = 20  # Keep last 20 messages

# ====== OPENROUTER API CONFIGURATION ======
# Get your free API key from: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
# Using OpenRouter's free Claude 3 Haiku model (fast and capable)
# You can change to other models: "anthropic/claude-3-haiku", "google/gemini-2.0-flash", "meta-llama/llama-3.2-3b-instruct", etc.
OPENROUTER_MODEL = "anthropic/claude-3-haiku"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

USE_REAL_AI = True

def construct_openrouter_headers():
    """Construct headers for OpenRouter API"""
    if not OPENROUTER_API_KEY:
        return None
    
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",  # Your site URL
        "X-Title": "AI Learning Assistant"  # Your app name
    }

def get_openrouter_response(question):
    """Get real AI response from OpenRouter API"""
    if not OPENROUTER_API_KEY:
        return "❌ API Key Error: No OpenRouter API key configured. Please set the OPENROUTER_API_KEY environment variable."
    
    print(f"[AI] Getting response for: {question[:50]}...")
    
    headers = construct_openrouter_headers()
    if not headers:
        return "❌ API configuration error. Please check your API key."
    
    # System instruction for the AI tutor
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
    
    # OpenRouter uses OpenAI-compatible API format
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
    
    # Retry configuration for rate limits
    max_retries = 3
    base_delay = 2  # Start with 2 seconds
    
    for attempt in range(max_retries):
        try:
            print(f"[AI] Sending request to OpenRouter API (Attempt {attempt + 1}/{max_retries})...")
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=data,
                timeout=45
            )
            
            # Handle rate limiting (429 error)
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"[AI] Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "⚠️ Rate limit exceeded. Please wait a moment and try again."
            
            # Handle other HTTP errors
            if response.status_code == 401:
                return "❌ Invalid API key. Please check your OpenRouter API key."
            elif response.status_code == 402:
                return "❌ Payment required. You may need to add credits to your OpenRouter account."
            elif response.status_code == 400:
                error_detail = response.json().get('error', {}).get('message', 'Bad request')
                return f"⚠️ Bad request: {error_detail}"
            
            response.raise_for_status()
            result = response.json()
            
            # Extract response from OpenRouter's format
            ai_response = result["choices"][0]["message"]["content"]
            print(f"[AI] Received response ({len(ai_response)} characters)")
            
            # Log token usage if available
            if "usage" in result:
                prompt_tokens = result["usage"].get("prompt_tokens", 0)
                completion_tokens = result["usage"].get("completion_tokens", 0)
                print(f"[AI] Token usage: {prompt_tokens} + {completion_tokens} = {prompt_tokens + completion_tokens}")
            
            return ai_response
            
        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:  # Last attempt
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
1. Get a free OpenRouter API key from: https://openrouter.ai/keys
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
        "ai_mode": f"OpenRouter ({OPENROUTER_MODEL})" if USE_REAL_AI and api_configured else "Mock Mode",
        "model": OPENROUTER_MODEL,
        "api_key_status": "configured" if api_configured else "not_configured",
        "api_key_preview": key_preview,
        "conversation_memory": len(conversation_history),
        "endpoints": {
            "health": "GET /health",
            "ask": "POST /ask",
            "history": "GET /history?limit=10",
            "config": "GET /config",
            "clear": "POST /clear",
            "models": "GET /models"
        }
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Main endpoint for asking questions"""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON or no data provided"}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        print(f"[REQUEST] Question: {question}")
        
        # Store in history with timestamp
        conversation_history.append({
            "role": "user",
            "content": question,
            "timestamp": time.strftime("%H:%M:%S")
        })
        
        # Trim history if too long
        if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
            del conversation_history[:len(conversation_history) - MAX_HISTORY_LENGTH * 2]
        
        # Get AI response
        start_time = time.time()
        
        if USE_REAL_AI and OPENROUTER_API_KEY:
            ai_response = get_openrouter_response(question)
            model_used = f"OpenRouter ({OPENROUTER_MODEL})"
        else:
            # Simulate thinking time based on question length
            think_time = min(1.5, 0.5 + (len(question) / 100))
            time.sleep(think_time)
            ai_response = generate_mock_response(question)
            model_used = "Mock AI"
        
        response_time = round(time.time() - start_time, 2)
        
        # Store AI response
        conversation_history.append({
            "role": "assistant",
            "content": ai_response,
            "model": model_used,
            "response_time": response_time,
            "timestamp": time.strftime("%H:%M:%S")
        })
        
        return jsonify({
            "answer": ai_response,
            "model_used": model_used,
            "response_time": f"{response_time}s",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "characters": len(ai_response),
            "success": True
        })
        
    except Exception as e:
        error_msg = f"Server error: {type(e).__name__}: {str(e)}"
        print(f"[SERVER ERROR] {error_msg}")
        return jsonify({
            "error": "An internal server error occurred",
            "details": str(e),
            "success": False
        }), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        limit = int(request.args.get('limit', 10))
        limit = min(limit, MAX_HISTORY_LENGTH)
    except ValueError:
        limit = 10
    
    return jsonify({
        "history": conversation_history[-limit:],
        "total_messages": len(conversation_history),
        "limit": limit,
        "max_history": MAX_HISTORY_LENGTH,
        "memory_usage": f"{len(conversation_history)}/{MAX_HISTORY_LENGTH * 2} messages"
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    api_configured = bool(OPENROUTER_API_KEY)
    
    return jsonify({
        "use_real_ai": USE_REAL_AI,
        "ai_provider": "OpenRouter",
        "model": OPENROUTER_MODEL,
        "api_key_configured": api_configured,
        "api_key_length": len(OPENROUTER_API_KEY) if OPENROUTER_API_KEY else 0,
        "max_history_length": MAX_HISTORY_LENGTH,
        "status": "ready" if (USE_REAL_AI and api_configured) else "setup_required",
        "setup_instructions": "Set OPENROUTER_API_KEY environment variable to enable real AI",
        "rate_limits": "Free tier available with limits"
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Get available OpenRouter models"""
    available_models = [
        {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku", "description": "Fast & capable (recommended)"},
        {"id": "google/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "description": "Google's fast model"},
        {"id": "meta-llama/llama-3.2-3b-instruct", "name": "Llama 3.2 3B", "description": "Meta's lightweight model"},
        {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "OpenAI's fast model"},
        {"id": "mistralai/mistral-7b-instruct", "name": "Mistral 7B", "description": "Good balance of speed/quality"}
    ]
    
    return jsonify({
        "available_models": available_models,
        "current_model": OPENROUTER_MODEL,
        "change_instruction": "To change model, update OPENROUTER_MODEL variable in code"
    })

@app.route('/clear', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return jsonify({
        "message": "Conversation history cleared",
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cleared_count": 0
    })

@app.route('/status', methods=['GET'])
def status():
    """Quick status check"""
    return jsonify({
        "online": True,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requests_served": len([m for m in conversation_history if m["role"] == "user"]),
        "ai_enabled": USE_REAL_AI and bool(OPENROUTER_API_KEY)
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🤖 AI LEARNING ASSISTANT - BACKEND SERVER v5.0")
    print("=" * 70)
    print(f"🌐 AI Provider: OpenRouter")
    print(f"🧠 AI Model: {OPENROUTER_MODEL}")
    print("-" * 70)
    
    # Check configuration
    api_configured = bool(OPENROUTER_API_KEY)
    key_preview = f"{OPENROUTER_API_KEY[:15]}..." if api_configured and len(OPENROUTER_API_KEY) > 15 else "Not set"
    
    if USE_REAL_AI and api_configured:
        print("✅ MODE: REAL AI (OpenRouter)")
        print(f"✅ API Key: Configured ({key_preview})")
        print("💡 Students will get real AI-powered answers!")
        print("📊 Using Claude 3 Haiku - fast and capable")
    elif USE_REAL_AI and not api_configured:
        print("⚠️ MODE: REAL AI (BUT API KEY NEEDED)")
        print("❌ API Key: NOT configured")
        print("\n📝 SETUP INSTRUCTIONS:")
        print("   1. Get FREE API key: https://openrouter.ai/keys")
        print("   2. Set environment variable:")
        print("      Windows CMD:  set OPENROUTER_API_KEY=your_key")
        print("      PowerShell:   $env:OPENROUTER_API_KEY=\"your_key\"")
        print("      Mac/Linux:    export OPENROUTER_API_KEY=your_key")
        print("   3. Restart server in SAME terminal")
        print("\n💰 OpenRouter offers free credits for new users!")
    else:
        print("🔧 MODE: MOCK AI (Testing)")
        print("💡 Set USE_REAL_AI = True to enable real AI")
    
    print("\n" + "=" * 70)
    print("🌐 SERVER ENDPOINTS:")
    print("   Health:  http://localhost:5000/health")
    print("   Ask:     POST http://localhost:5000/ask")
    print("   History: http://localhost:5000/history?limit=10")
    print("   Config:  http://localhost:5000/config")
    print("   Models:  http://localhost:5000/models")
    print("   Status:  http://localhost:5000/status")
    print("=" * 70)
    print("\n🚀 Starting server on http://localhost:5000")
    print("🛑 Press CTRL+C to stop")
    print("=" * 70)
    
    app.run(debug=True, port=5000, threaded=True)