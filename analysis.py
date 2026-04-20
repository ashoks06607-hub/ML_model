import os
from google import genai
from dotenv import load_dotenv
load_dotenv()

key=os.getenv('google_api_key')
genai.Client(api_key=key)

model=genai.GenerativeModel('gemini-2.5-flash-lite')

def generate_code(results_df):
    prompt=f'''You are a data scientist expert. Here are a few rows of a dataset and the results of some machine learning models on that dataset. Please analyze the results and suggest which model is best for this dataset and why.1. identify the best model, 2. explain why it is the best model, 3. suggest any improvements that can be made to the models or the dataset to achieve better results.'''
    response=model.generate_content(prompt)
    return response.text 

def suggest_improvements(results_df):
    prompt=f'''You are a data scientist expert. Here are the model results: 
    {results_df.to_string()}
    Suggestion: 
    - Identify the best model based on the results.
    - Explain why it is the best model.
    - Suggest any improvements that can be made to the models or the dataset to achieve better results.'''
    response=model.generate_content(prompt)
    return response.text