# RAG Agent 离线评测

##目标

验证 RAG Agent 能否从本地知识库中找到正确答案，并控制工具调用伦茨和响应耗时。

## 文件结构

- `cases.yaml`：评测问题、答案关键词和最大工具轮数
- `metrics.py`：答案匹配、工具轮数和完成率等指标
- `run_eval.py`：执行评测并生成报告
- `report.json`：最近一次评测结果

## 运行方式

在项目根目录执行：

```powershell
$env:PYTHONPATH = "src"
python -m evals.run_eval