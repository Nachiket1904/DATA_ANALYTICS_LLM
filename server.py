from flask import Flask, request, jsonify
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import random
import logging
from app import app  # Assuming 'app' is the Flask instance in app.py




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


# Create Flask app instance
app = Flask(__name__)

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

# Root endpoint
@app.route('/')
def read_root():
    return jsonify({"message": "Welcome to my API!"})

# Define the request model
class GeneratePromptsRequest(BaseModel):
    content_type: str
    audience_type: str
    delivery_method: str
    content_theme: str
    target_industry: str

class PromptRequest(BaseModel):
    prompt: str

# Endpoint to generate content creation prompts
@app.route('/generate_prompts', methods=['POST'])
def generate_prompts():
    try:
        data = request.get_json()  # Get JSON from request
        request_data = GeneratePromptsRequest(**data)  # Validate and parse the request data

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
                        \nContent Type - {request_data.content_type}\nAudience Type - {request_data.audience_type}\nDelivery Method - {request_data.delivery_method}
                        \nContent Theme - {request_data.content_theme}\nTarget Industry - {request_data.target_industry}
                        \nPlease format your response as a JSON array."""
                    ],
                }
            ]
        )
        response = chat_session.send_message("Generate the content creation prompts.")
        prompts = extract_numbered_points(response.text)

        if not prompts:
            return jsonify({"error": "Error extracting prompts. Please check the API response format."}), 400

        return jsonify({"prompts": prompts})

    except Exception as e:
        return jsonify({"error": f"Error generating prompts: {e}"}), 500

# Endpoint to ask a specific prompt to Gemini API
@app.route('/ask-gemini/', methods=['POST'])
def ask_gemini():
    try:
        data = request.get_json()  # Get JSON from request
        prompt_request = PromptRequest(**data)  # Validate and parse the request data

        selected_prompt = prompt_request.prompt

        # Send the selected prompt to Gemini
        chat_session = genai.GenerativeModel(
            model_name="gemini-1.5-pro"
        ).start_chat(
            history=[{
                "role": "user",
                "parts": [selected_prompt]
            }]
        )

        response = chat_session.send_message(selected_prompt)
        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
