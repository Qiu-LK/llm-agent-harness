from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
    return events


def _e(value: Any) -> str:
    return html.escape(str(value))


def render_html(events: list[dict[str, Any]], title: str = "Harness Run") -> str:
    manifest = next((e for e in events if e.get("event") == "run_manifest"), {})
    finished = next((e for e in reversed(events) if e.get("event") == "run_finished"), {})
    failed = next((e for e in reversed(events) if e.get("event") in {"run_failed", "run_error"}), {})
    status = "成功" if finished else "失败" if failed else "未完成"
    duration = max((float(e.get("elapsed_ms", 0)) for e in events), default=0)
    steps = max((int(e.get("step", 0)) for e in events), default=0)
    labels = {"run_started": "开始任务", "model_request": "请求模型", "model_response": "模型响应", "tool_started": "调用工具", "tool_finished": "工具结果", "run_finished": "完成", "run_failed": "失败", "run_error": "错误", "run_manifest": "运行配置"}
    cards = []
    for event in events:
        kind = str(event.get("event", "event"))
        payload = {k: v for k, v in event.items() if k not in {"event", "timestamp", "run_id", "elapsed_ms"}}
        cards.append(f'<article class="event {kind}"><div class="dot"></div><div><h3>{_e(labels.get(kind, kind))}<span>{_e(event.get("elapsed_ms", "?"))} ms</span></h3><pre>{_e(json.dumps(payload, ensure_ascii=False, indent=2, default=str))}</pre></div></article>')
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title><style>
:root{{--bg:#0b1220;--panel:#111c2f;--line:#2a3a55;--text:#e7eefb;--muted:#8ca0bd;--accent:#55c2ff;--ok:#4ade80;--bad:#fb7185}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(135deg,#07101d,#101b31);color:var(--text);font:14px system-ui}}main{{max-width:1050px;margin:auto;padding:36px 18px}}h1{{font-size:30px;margin:0 0 20px}}.summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:28px}}.metric{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}}.metric b{{display:block;font-size:20px;margin-top:5px}}.metric small{{color:var(--muted)}}.timeline{{position:relative;margin-left:12px}}.timeline:before{{content:"";position:absolute;left:7px;top:0;bottom:0;width:2px;background:var(--line)}}.event{{display:grid;grid-template-columns:28px 1fr;position:relative;margin:0 0 14px}}.dot{{width:16px;height:16px;border:3px solid var(--bg);border-radius:50%;background:var(--accent);z-index:1;margin-top:17px}}.run_finished .dot{{background:var(--ok)}}.run_failed .dot,.run_error .dot{{background:var(--bad)}}.event>div:last-child{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;min-width:0}}h3{{margin:0;color:#cfeaff}}h3 span{{float:right;color:var(--muted);font-size:12px;font-weight:400}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:10px 0 0;color:#cbd8eb;max-height:420px;overflow:auto}}
</style></head><body><main><h1>{_e(title)}</h1><section class="summary"><div class="metric"><small>状态</small><b>{_e(status)}</b></div><div class="metric"><small>模型</small><b>{_e(manifest.get('model','-'))}</b></div><div class="metric"><small>步骤</small><b>{steps}</b></div><div class="metric"><small>耗时</small><b>{duration:.0f} ms</b></div><div class="metric"><small>Token</small><b>{_e(finished.get('total_tokens','-'))}</b></div></section><section class="timeline">{''.join(cards)}</section></main></body></html>"""


def write_report(trace_path: Path, output_path: Path | None = None) -> Path:
    output = output_path or trace_path.with_suffix(".html")
    output.write_text(render_html(load_events(trace_path), trace_path.stem), encoding="utf-8")
    return output


def write_eval_report(results: Iterable[Any], output: Path) -> Path:
    rows = list(results)
    passed = sum(bool(r.passed) for r in rows)
    total = len(rows)
    avg_time = sum(float(r.duration_ms) for r in rows) / total if total else 0
    avg_steps = sum(int(r.steps) for r in rows) / total if total else 0
    table = "".join(f"<tr><td>{_e(r.case_id)}</td><td>{_e(r.category)}</td><td class={'ok' if r.passed else 'bad'}>{'PASS' if r.passed else 'FAIL'}</td><td>{r.steps}</td><td>{r.total_tokens}</td><td>{r.duration_ms:.0f} ms</td><td>{_e('; '.join(r.failures))}</td></tr>" for r in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Evaluation Report</title><style>body{{font:14px system-ui;background:#f4f7fb;color:#172033;margin:32px}}.stats{{display:flex;gap:14px}}.card{{background:white;padding:18px;border-radius:12px;box-shadow:0 3px 18px #ccd6e5}}table{{width:100%;border-collapse:collapse;background:white;margin-top:24px}}th,td{{padding:12px;border-bottom:1px solid #e2e8f0;text-align:left}}.ok{{color:#15803d;font-weight:bold}}.bad{{color:#be123c;font-weight:bold}}</style></head><body><h1>自动评测报告</h1><div class="stats"><div class="card">通过率<br><b>{passed}/{total} ({passed/total*100 if total else 0:.1f}%)</b></div><div class="card">平均耗时<br><b>{avg_time:.0f} ms</b></div><div class="card">平均步骤<br><b>{avg_steps:.1f}</b></div></div><table><thead><tr><th>用例</th><th>分类</th><th>状态</th><th>步骤</th><th>Token</th><th>耗时</th><th>失败原因</th></tr></thead><tbody>{table}</tbody></table></body></html>""", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a Harness JSONL trace to HTML")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(write_report(args.trace, args.output).resolve())
