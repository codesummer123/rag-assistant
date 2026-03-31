import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL=os.getenv("OPENAI_MODEL")

REWRITE_PROMPT = """你是一个搜索专家。请将用户的原始问题改写成 1-2 个最适合在向量数据库中检索的关键词或短句。
直接输出改写后的内容，不要有任何解释。

示例：
输入：那个请假怎么整？
输出：年假申请流程 员工年假规定

输入：{query}
输出："""

def rewrite_query(query: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()