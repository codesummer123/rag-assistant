import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("OPENAI_MODEL")

RERANK_PROMPT = """你是一个相关性评价专家。
请根据用户的问题，对提供的参考资料片段进行相关性排序。
只返回最相关的片段的索引编号（从0开始），按相关度从高到低排列，用逗号分隔。

用户问题：{query}

参考资料：
{contexts}

请输出排序后的索引列表（例如：2,0,1）："""

def rerank(query: str, contexts: list[dict], top_n: int = 3) -> list[dict]:
    if not contexts:
        return []

    context_str = ""
    for i, c in enumerate(contexts):
        context_str += f"索引[{i}]: {c['text'][:200]}\n\n"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": RERANK_PROMPT.format(query=query, contexts=context_str)}],
        temperature=0
    )

    try:
        indices_str = response.choices[0].message.content.strip()
        order = [int(idx.strip()) for idx in indices_str.split(",") if idx.strip().isdigit()]
        reranked = [contexts[i] for i in order if i < len(contexts)]
        return reranked[:top_n]
    except Exception as e:
        print(f"[rerank] 解析失败: {e}")
        return contexts[:top_n]