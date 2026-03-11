from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# How to get your Databricks token: https://docs.databricks.com/en/dev-tools/auth/pat.html
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
DATABRICKS_BASE_URL = os.environ.get('DATABRICKS_BASE_URL')

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=DATABRICKS_BASE_URL,
)

response = client.chat.completions.create(
    model="databricks-claude-opus-4-6",
    messages=[
        {
            "role": "user",
            "content": "What is an LLM agent?"
        }
    ],
    max_tokens=5000
)

print(response.choices[0].message.content)