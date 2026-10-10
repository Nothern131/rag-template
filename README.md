# 通用 RAG 模板

面向 200-1500 人规模游戏公司的**客服/知识库 RAG 试点**脚手架，3 天可交付可用版本。

> 需要 3 天交付 / 调参 / 验收服务？见 [rag-delivery-repo](https://github.com/Nothern131/rag-delivery-repo)（销售页 + 定价 + 8 层验收清单）

## 3 步上手

### 1. 装环境
```bash
pip install -r requirements.txt
```

### 2. 放数据
把 3 份文档丢进 `data/` 目录：
- `faq.txt` / `faq.md` / `faq.csv` 等（支持 txt / md / csv / json / jsonl / html）

### 3. 启动
```bash
# 演示界面（推荐）
python app.py

# 或 API 服务
python server.py
```

默认用 **本地嵌入 + Mock LLM**，无需 API key。要接真 LLM：

```bash
# 复制 .env.example → .env，填 OPENAI_API_KEY，然后
export LLM_BACKEND=openai
export EMBEDDING_BACKEND=openai   # 可选
python server.py
```

## 目录结构
```
rag-template/
  config.py        所有可调参数（切分/嵌入/LLM/检索/服务）
  loader.py        多格式文档加载
  chunker.py       语义感知切分 + 重叠
  embedder.py      本地 / OpenAI 嵌入，自动降级
  store.py         Chroma 持久化 + 健康检查
  llm.py           OpenAI / Anthropic / 本地 / Mock 四后端
  rag.py           检索 + 生成 + 溯源 + 置信度
  server.py        FastAPI（/ask, /ask/stream, /ingest, /health, /metrics）
  app.py           Gradio 演示界面
  tests/           6 个核心测试（Mock 后端，无网络）
  data/            放你的文档
```

## 定制点（客户化改这里）

| 需求 | 改哪个文件 |
|---|---|
| 换数据源（接 DB / API） | `loader.py` 加一个 `load_db()` |
| 换向量库 | `store.py` 换成 Milvus / Qdrant / pgvector |
| 加业务规则（意图分类 / 拒答） | `llm.py` 的 `SYSTEM_PROMPT` |
| 加飞书/钉钉机器人入口 | 新增 `bot.py`，复用 `rag.answer()` |
| 加鉴权 | `server.py` 加 middleware |
| 换 LLM（DeepSeek / 通义 / Kimi） | `llm.py` 加一个 Backend 类 |

## 测试
```bash
pytest tests/ -v
```

## 质量门（已满足 8 层约束）
- [x] 异常处理：所有外部调用 try/catch，返回结构统一
- [x] 幻觉防御：`openai`/`anthropic` 延迟导入，不编造 API
- [x] 可观测性：`/health` + `/metrics` + 结构化日志
- [x] 韧性：嵌入失败自动降级到本地，LLM 指数退避重试
- [x] 资源：`chroma` 持久化目录可清理，文档按块 upsert 不重复
- [x] 类型：所有函数带类型标注，禁止 any
- [x] 安全：API key 走环境变量，无硬编码
- [x] 边界：空问题 / 无匹配 / 超长问题 都有处理
