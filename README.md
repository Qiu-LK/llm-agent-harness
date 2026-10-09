# LLM Agent Harness v0.3

一个适合作品集和学习用途的 Python Agent Harness。它可以连接 DeepSeek、Ollama 或其他 Chat Completions 兼容服务，分析本地代码仓库、实时展示工具调用、自动运行评测，并以人工审批方式应用代码修改。

## 功能

- 模型 → 工具 → 模型的 Agent 循环
- DeepSeek、通用兼容接口、Ollama 三种模型配置
- `list_files`、`read_file`、`search_text` 只读工具
- `propose_edit` 受控修改工具：只生成 diff，必须由用户批准才写入
- 文件哈希冲突保护，避免覆盖提案生成后的用户修改
- FastAPI + SSE 实时 Web 执行时间线
- 自动执行 JSONL 评测用例并统计通过率、步骤、耗时和 token
- JSONL 轨迹、可视化时间线报告和评测汇总报告
- 超时、重试、路径隔离、重复工具调用检测

## 安装

需要 Python 3.11 或更高版本：

```powershell
cd "C:\Users\邱璟豪\Desktop\新建文件夹"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

## 模型配置

DeepSeek：

```env
MODEL_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

Ollama：

```env
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
```

其他兼容服务：

```env
MODEL_PROVIDER=compatible
MODEL_BASE_URL=https://example.com/v1
MODEL_NAME=模型名称
MODEL_API_KEY=服务密钥
```

## CLI

```powershell
deepseek-harness "阅读项目并说明启动流程" --workspace "D:\你的项目"
deepseek-harness "改进错误处理，并提出修改" --workspace "D:\你的项目" --allow-edit-proposals
```

CLI 只创建提案，不会自动批准或写入。代码审批建议使用 Web 页面。

## Web 界面

```powershell
deepseek-web
```

浏览器访问 `http://127.0.0.1:8000`。页面支持输入任务与工作区、SSE 实时事件、查看 diff，以及明确批准或拒绝修改提案。该服务面向本机演示，不应直接暴露到公网。

## 自动评测闭环

项目自带 24 条固定评测任务，覆盖项目结构、入口分析、配置读取、函数理解、测试分析、文档核对和问题发现。用例文件每行一条 JSON：

```json
{"id":"project-summary","prompt":"阅读项目并找到入口文件","workspace":"fixtures/sample_project","expected_contains":["main.py"],"expected_tools":["list_files","read_file"],"max_steps":6,"no_tool_errors":true}
```

```powershell
deepseek-eval evals\cases.example.jsonl --output eval-results\latest
```

评测器会逐条调用模型、保存轨迹、执行断言、输出 `results.json`，并生成 `report.html`。这一步会使用所配置模型并产生相应 API 用量。

### 真实基准结果

2026-10-03 使用 `deepseek-chat` 运行 24 条固定用例：

| 指标 | 结果 |
|---|---:|
| 通过用例 | 22 / 24 |
| 成功率 | 91.7% |
| 平均耗时 | 3.53 秒 |
| 总 Token | 84,438 |

完整报告位于 `eval-results/real-2026-10-03/report.html`。两条失败均来自严格的答案关键词断言，运行过程没有崩溃或工具错误。

## 前端结构

Web 页面已从 Python 后端拆分：

```text
src/deepseek_harness/
├── web.py
└── web_assets/
    ├── templates/index.html
    └── static/
        ├── app.css
        └── app.js
```

`web.py` 只保留 API、SSE 和会话管理，页面结构、样式与交互分别维护。

## 单次运行报告

```powershell
deepseek-report "runs\<run-id>.jsonl"
```

报告包含状态、模型、步骤、耗时、token 汇总和完整执行时间线。

## 测试

```powershell
python -m unittest discover -s tests -v
```

单元测试使用假模型或 Mock HTTP，不消耗 API 额度。

## 受控修改机制

```text
模型提交完整的新内容
        ↓
系统验证工作区路径
        ↓
生成 unified diff 和原文件 SHA-256
        ↓
等待用户批准
        ↓
再次核对文件哈希
        ↓
应用或因冲突而拒绝
```

`propose_edit` 只能修改已存在的 UTF-8 文件。当前没有任意 Shell、自动测试执行和文件删除能力。这是面向单用户、本机演示的学生项目，不是完整安全沙箱。
