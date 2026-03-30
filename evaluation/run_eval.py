import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from rag.retriever import retrieve
from rag.generator import generate

QUESTIONS_PATH  = "evaluation/questions.json"
REPORT_PATH = "evaluation/report.md"

def run_eval():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        start = time.time()
        contexts = retrieve(q["query"], k=3)
        answer = generate(q["query"], contexts)
        latency = int((time.time() - start) * 1000)

        top_source = contexts[0]["source"] if contexts else None
        source_hit = {top_source == q["expected_source"]} if q["expected_source"] else None
        has_citations = len(contexts) > 0

        results.append({
            "id": q["id"],
            "query": q["query"],
            "answer": answer,
            "top_source": top_source,
            "expected": q["expected_source"],
            "source_hit": source_hit,
            "has_citations": has_citations,
            "latency_ms": latency,
        })

        status = "✅" if source_hit else ("⚪" if source_hit is None else "❌")
        print(f"[{status}] Q{q['id']}: {q['query'][:20]}... ({latency}ms)")

    write_report(results)
    print(f"报告已写入 {REPORT_PATH}")

def write_report(results: list[dict]):
    total = len(results)
    scorable = [r for r in results if r["source_hit"] is not None]
    hits = [r for r in results if r["source_hit"]]
    hit_rate = len(hits) / len(scorable) * 100 if len(scorable) > 0 else 0
    avg_lat = sum(r["latency_ms"] for r in results) / total

    lines = [
        "# 评估报告\n",
        f"- 测试问题：{total}",
        f"- 来源命中：{len(hits)/len(scorable)} = {hit_rate:.0f}%",
        f"- 平均响应：{avg_lat:.0f} ms\n",
        "## 详情结果\n"
        "| ID | 问题 | Top来源 | 期望来源 | 命中 | 耗时(ms) |",
        "|----|-----|--------|---------|------|---------|",
    ]

    for r in results:
        hit_str = "✅" if r["source_hit"] else ("⚪" if r["source_hit"] is None else "❌")
        lines.append(
            f"| {r['id']} | {r['query'][:18]} | "
            f"{r['top_source'] or '-'} | "
            f"{r['expected'] or '(拒答)'} | "
            f"{hit_str} | {r['latency_ms']} |"
        )

    lines += ["\n## 答案明细\n"]
    for r in results:
        lines.append(f"### Q{r['id']}：{r['query']}\n")
        lines.append(f"**答案**：{r['answer']}\n")
        lines.append(f"**来源**：{r['top_source']}  \n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    run_eval()