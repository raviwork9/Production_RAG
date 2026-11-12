import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
key = os.environ["GROQ_API_KEY"].strip().strip('"').strip("'")
print("key length:", len(key), "| starts with:", key[:4])

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
out = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(out.choices[0].message.content)