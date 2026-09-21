"""
Gradio 演示界面 — 一键启动。
"""
from __future__ import annotations

import gradio as gr

from config import load as load_config
from rag import RAGService


def build_ui() -> gr.Blocks:
    cfg = load_config()
    svc = RAGService(cfg)

    def on_ingest() -> str:
        res = svc.ingest_directory()
        return f"已索引 {res['docs']} 份文档 / {res['chunks']} 个片段，耗时 {res['elapsed_ms']}ms"

    def on_ask(question: str, top_k: int) -> tuple[str, str, str, float]:
        ans = svc.answer(question, top_k=int(top_k))
        citations = "\n".join(
            f"[{i+1}] {c.doc_path}#{c.index} (score={c.score:.3f})\n    {c.text[:200]}"
            for i, c in enumerate(ans.citations)
        )
        err = f"\n\n⚠️ 错误: {ans.error}" if ans.error else ""
        return ans.answer + err, citations, f"模型={ans.model} 延迟={ans.latency_ms}ms", ans.confidence

    def on_stream(question: str) -> str:
        parts: list[str] = []
        for tok in svc.stream_answer(question):
            parts.append(tok)
            gr.skip()
        return "".join(parts)

    with gr.Blocks(title="RAG 演示") as blocks:
        gr.Markdown("# 通用 RAG 演示")
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="问题", placeholder="输入问题…")
                with gr.Row():
                    k = gr.Slider(1, 10, 5, step=1, label="top_k")
                    ask_btn = gr.Button("提问")
                    stream_btn = gr.Button("流式提问")
                answer_box = gr.Textbox(label="回答", interactive=False)
                with gr.Row():
                    conf = gr.Number(label="置信度", interactive=False)
                    meta = gr.Textbox(label="元信息", interactive=False)
                citations = gr.Textbox(label="引用来源", lines=8, interactive=False)
            with gr.Column(scale=1):
                ingest_btn = gr.Button("重新索引 data/ 目录")
                status = gr.Textbox(label="状态", interactive=False)

        ingest_btn.click(on_ingest, None, status)
        ask_btn.click(on_ask, [q, k], [answer_box, citations, meta, conf])
        stream_btn.click(on_ask, [q, k], [answer_box, citations, meta, conf])

    return blocks


def main() -> None:
    blocks = build_ui()
    blocks.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
