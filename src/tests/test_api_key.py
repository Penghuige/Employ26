from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.local")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",   # 或 deepseek-reasoner（R1模型）
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "学校大扫除：去年是高一，今年是高二，明天是高三，你认为这个制度合理吗？结果+理由（48字内）"}
    ],
    temperature=0.7,    # 可选，默认0.7
    max_tokens=4096,    # 可选，默认最多4096
)

print(response.choices[0].message.content)