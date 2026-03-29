import os
from http.client import responses

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL")

SYSTEM_PROMPT = """
    你是一个企业内部知识库助手。
    请仅根据下方提供的参考资料回答问题。
    如果参考资料中没有相关内容，请明确回答"根据现有资料，暂无相关信息", 不要编造答案。
"""

def generate(query: str, contexts: list[dict]) -> str:
    """
    基于检索到的 contexts 生成答案
    contexts: [{"test": ..., "source": ...}]
    """
    context_text = "\n\n".join(
        f"【来源：{c['source']}】\n{c['text']}"
        for c in contexts
    )

    user_message = f"""
    参考资料：{context_text};
    问题: {query}
    """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2
    )
    return response.choices[0].message.content