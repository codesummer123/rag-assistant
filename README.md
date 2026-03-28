# CN Enterprise RAG Assistant

一个面向中文企业知识库场景的 RAG MVP。  
当前版本目标：先跑通 `索引 -> 检索 -> 生成` 的最小闭环。

## Features

- [x] FastAPI 服务
- [x] `/health` 健康检查
- [ ] 文档切分与索引（Day2）
- [ ] 向量检索（Day3）
- [ ] `/chat` RAG 问答（Day4）
- [ ] 最小评估（Day5）

## Project Structure

```text
app/            # API 层
rag/            # RAG 核心逻辑
data/kb/        # 知识库文档
evaluation/     # 评估数据与脚本
scripts/        # 辅助脚本