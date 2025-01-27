from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import google.generativeai as genai
import random
import json
import logging
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Function to randomly select an API key
def get_random_api_key():
    gemini_api_keys = os.getenv("GEMINI_API_KEYS")
    if gemini_api_keys:
        api_keys = gemini_api_keys.split(",")
        selected_key = random.choice(api_keys)  # Randomly select one key
        logging.info(f"Selected API key: {selected_key}")  # Log the selected key
        return selected_key
    else:
        raise ValueError("GEMINI_API_KEYS not set in the .env file.")

# Configure Gemini API with a random API key
api_key = get_random_api_key()
genai.configure(api_key=api_key)
print(f"Using API key: {api_key}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Function to extract numbered points from response
def extract_numbered_points(text):
    try:
        json_start = text.find('```json\n[')
        json_end = text.find(']\n```')
        if json_start != -1 and json_end != -1:
            json_str = text[json_start + 7:json_end + 1]
            return json.loads(json_str)
        else:
            return [text.strip()]
    except json.JSONDecodeError as e:
        return [f"Malformed JSON: {e}"]
    except Exception as e:
        return [f"Unexpected error: {e}"]

# Default route
@app.route("/")
def home():
    return jsonify({"message": "Welcome to my API!"})

# Endpoint to generate content creation prompts
@app.route("/generate_prompts", methods=["POST"])
def generate_prompts():
    try:
        data = request.json
        # Validate input fields
        required_fields = ["content_type", "audience_type", "delivery_method", "content_theme", "target_industry"]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields in the request."}), 400

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
            system_instruction="You are expert at prompt engineering and your goal is to write prompts helping the trainers to create professional and relevant content."
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"""Generate 4 content creation prompts to help trainers generate content based on the following inputs:
                        \nContent Type - {data['content_type']}\nAudience Type - {data['audience_type']}\nDelivery Method - {data['delivery_method']}
                        \nContent Theme - {data['content_theme']}\nTarget Industry - {data['target_industry']}
                        \nPlease format your response as a JSON array."""
                    ],
                }
            ]
        )
        response = chat_session.send_message("Generate the content creation prompts.")
        prompts = extract_numbered_points(response.text)

        if not prompts:
            return jsonify({"error": "Error extracting prompts. Please check the API response format."}), 500

        return jsonify({"prompts": prompts})
    except Exception as e:
        return jsonify({"error": f"Error generating prompts: {e}"}), 500

# Endpoint to ask a specific prompt to Gemini API
@app.route("/ask-gemini", methods=["POST"])
def ask_gemini():
    try:
        data = request.json
        print(data)
        # Validate the input
        if "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in the request."}), 400

        selected_prompt = data["prompt"]
        print(selected_prompt)

        # Send the selected prompt to Gemini
        chat_session = genai.GenerativeModel(
            model_name="gemini-1.5-pro"
        ).start_chat(
            history=[{
                "role": "user",
                "parts": [selected_prompt]
            }]
        )
        print(chat_session)

        response = chat_session.send_message(selected_prompt)
        print(response)
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
