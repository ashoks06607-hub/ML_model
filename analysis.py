import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=key)


def generate_code(results_df):
    prompt = f"""
You are a data scientist expert.

Dataset results:
{results_df.head(10).to_string()}

Tasks:
1. Identify best model
2. Explain why
3. Suggest improvements
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"


def suggest_improvements(results_df):
    prompt = f"""
Model results:
{results_df.head(10).to_string()}

Give:
- Best model
- Reason
- Improvements
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"