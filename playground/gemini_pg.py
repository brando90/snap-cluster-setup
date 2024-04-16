import google.generativeai as genai
import os

# Function to configure API Key
def configure_api_key(api_key):
    genai.configure(api_key=api_key)

# Function to generate content using Gemini Pro
def generate_content(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text

# Main execution
if __name__ == "__main__":
    api_key = os.getenv('GOOGLE_API_KEY')
    configure_api_key(api_key)

    # Example prompt
    prompt = "What is the meaning of life?"
    response_text = generate_content(prompt)
    print("Generated response:", response_text)

