# 模板定制指南（改这 3 处）

## 1. 换数据源（接数据库 / API）

**改** `loader.py`，加一个函数：

```python
# loader.py 末尾加
def load_from_db(sql: str, db_path: str) -> List[Doc]:
    """从 SQLite 读取问答对"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute(sql).fetchall()
    docs = []
    for i, (q, a) in enumerate(rows):
        text = f"Q: {q}\nA: {a}"
        docs.append(Doc(path=f"db_row_{i}", text=text, source_type="sql", metadata={}))
    return docs
```

然后在 `rag.py` 里 `ingest_directory()` 之后加一个 `ingest_db(db_path)` 方法。

---

## 2. 换 LLM（DeepSeek / 通义 / Kimi）

**改** `llm.py`，加一个 Backend 类：

```python
class DeepSeekBackend:
    def __init__(self, cfg):
        from openai import OpenAI
        self._client = OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url="https://api.deepseek.com/v1",
        )
        self._model = cfg.deepseek_model  # deepseek-chat 或 deepseek-reasoner

    def chat(self, question, context_blocks):
        prompt = build_prompt(question, context_blocks)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.llm_max_tokens,
        )
        return LLMResult(resp.choices[0].message.content, self._model, "deepseek", 0)
```

然后在 `config.py` 加一行 `deepseek_api_key = ""`，在 `build_backend()` 加 `elif cfg.llm_backend == "deepseek"`。

---

## 3. 加飞书机器人入口

**新建** `bot.py`：

```python
from rag import RAGService

svc = RAGService.load()  # 从 config 加载

def on_message(chat_id, question):
    ans = svc.answer(question)
    return ans.answer, ans.citations
```

然后在飞书机器人后台配置 webhook → 这个函数。

---

## 4. 常见客户化场景

| 行业 | 改什么 | 时间 |
|---|---|---|
| 游戏 | 不改，直接用 | 0h |
| 金融投研 | 换数据源（接 wind/iFind）+ 加风控拒答规则 | 1 天 |
| 法律 | 换 LLM（法律专用模型）+ 加案例检索模式 | 2 天 |
| 医疗 | 换数据源（接医院系统）+ 强隐私隔离 | 3 天 |
| 教育 | 加多轮对话 + 学生错题本 | 2 天 |

---

## 5. 部署到客户服务器

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 放数据
cp 客户的FAQ.md data/
cp 客户的SOP.md data/
cp 客户的玩家QA.csv data/

# 3. 配置 API key
cp .env.example .env
# 编辑 .env，填 OPENAI_API_KEY / ANTHROPIC_API_KEY

# 4. 跑
python app.py
```

---

*本模板基于公开开源库（Chroma / sentence-transformers / FastAPI / Gradio），MIT 协议可商用。*
