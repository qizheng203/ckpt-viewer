# Ckpt_viewer v2 中文说明

## 1. v2 已实现功能

Ckpt_viewer v2 是一个本地运行的 checkpoint 权重体检与静态量化风险诊断工具。

已实现：

- 支持 `.pth`、`.pt`、`.ckpt`、`.bin`、`.safetensors`。
- 支持 Web UI 和 CLI。
- 默认使用 PyTorch `weights_only=True` 安全加载。
- 支持可信文件的 unsafe load 兜底。
- 自动识别常见 `state_dict`。
- 区分 `weight`、`bias`、`norm_weight`、`norm_bias`、`running_mean`、`running_var`、`num_batches_tracked`、`buffer`。
- Overview 默认排除无关 buffer，避免 int64 buffer 污染图表。
- Tensor Table 支持 role、dtype、group、type、risk、fallback candidate 筛选。
- Layer Detail 支持基础统计、离群值、量化误差、通道指标、BN/Norm 指标和自动诊断。
- Quantization Risk 支持 per-tensor / per-channel INT8 对比、fallback candidates 和推荐部署策略。
- BN / Norm Risk 支持 running_var 过小和 norm gamma 异常检查。
- 支持导出 CSV / JSON / Markdown / HTML 报告。

## 2. 发布目录结构

```text
ckpt-viewer/
├── ckpt_viewer/
├── doc/
├── examples/
├── install.bat
├── install.ps1
├── install.sh
├── LICENSE
├── pyproject.toml
└── README.md
```

该目录已移除开发测试目录、开发规划文档、缓存文件和示例二进制权重，只保留用户安装与使用需要的内容。

## 3. 一键安装

Linux / macOS：

```bash
bash install.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Windows CMD：

```bat
install.bat
```

安装完成后，新打开任意终端即可运行：

```bash
ckpt-viewer
```

更多安装说明见：

```text
doc/一键安装与全局启动说明.md
```

## 4. 手动安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
ckpt-viewer
```

Windows：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
ckpt-viewer
```

## 5. Web UI 使用

```bash
ckpt-viewer
```

或：

```bash
ckpt-viewer ui
```

默认地址：

```text
http://localhost:8501
```

## 6. CLI 使用

基础分析：

```bash
ckpt-viewer analyze /path/to/model.pth --out report
```

导出完整报告：

```bash
ckpt-viewer analyze /path/to/model.pth --out report --export csv,json,md,html
```

只看可量化权重：

```bash
ckpt-viewer analyze /path/to/model.pth --out report --exclude-buffers --quantizable-only
```

安全加载失败且确认文件可信：

```bash
ckpt-viewer analyze /path/to/model.pth --trusted --out report
```

## 7. 示例

仓库不附带二进制示例权重。需要测试时可生成小 checkpoint：

```bash
python examples/create_tiny_checkpoint.py
ckpt-viewer analyze examples/tiny_checkpoint.pth --out report --export csv,json,md,html
```

## 8. 导出报告

```text
report/
├── summary.json
├── tensor_stats.csv
├── quant_stats.csv
├── bn_norm_stats.csv
├── recommendations.json
├── warnings.json
├── report.md
├── report.html
└── figures/
```

## 9. GitHub 发布

```bash
git init
git add .
git commit -m "Release Ckpt_viewer v2"
git branch -M main
git remote add origin https://github.com/<yourname>/ckpt-viewer.git
git push -u origin main
```

别人使用：

```bash
git clone https://github.com/<yourname>/ckpt-viewer.git
cd ckpt-viewer
bash install.sh
ckpt-viewer
```

Windows 用户：

```powershell
git clone https://github.com/<yourname>/ckpt-viewer.git
cd ckpt-viewer
powershell -ExecutionPolicy Bypass -File .\install.ps1
ckpt-viewer
```

## 10. 安全和限制

默认 safe load 会尽量避免执行 pickle 中的任意 Python 对象。不要对陌生来源的 `.pth` 文件启用 unsafe load。

Ckpt_viewer 当前只做静态权重分析，不分析 activation 分布，不执行模型 forward，不替代校准集和验证集评估。真实 PTQ / INT8 效果必须结合 calibration data、task metrics 和部署后 engine 验证。
