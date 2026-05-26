# Ckpt_viewer
## 如果您觉得本项目对您的工作有帮助,请奖励我们一个Star,感谢您!如果您觉得项目有需要完善的地方,请提交Issues,或者Email:qzheng203@gmail.com
Ckpt_viewer 是一个本地运行的 checkpoint 权重体检与静态量化风险诊断工具。它可以直接分析 `.pth`、`.pt`、`.ckpt`、`.bin`、`.safetensors` 文件中的 tensor 分布、离群值、稀疏性、INT8 fake quant 风险和 BN / Norm folding 风险。

本工具不需要 GPU，不需要模型源码，不执行 forward，也不需要输入样本。

## 主要功能

- 自动安全加载 PyTorch checkpoint，支持可信文件的 unsafe load 兜底。
- 自动识别常见 `state_dict` 结构。
- 区分 `weight`、`bias`、`norm_weight`、`norm_bias`、`running_mean`、`running_var`、`num_batches_tracked`、`buffer`。
- Overview 默认排除无关 buffer，避免 `num_batches_tracked` 污染图表。
- Tensor Table 支持 role、dtype、group、type、risk、fallback candidate 等筛选。
- Layer Detail 提供直方图、`log10(abs(weight)+eps)` histogram、percentile、per-channel 图表和自动诊断。
- Quantization Risk 给出 per-tensor / per-channel INT8 对比、fallback candidates 和推荐部署策略。
- BN / Norm Risk 提示 Conv-BN folding 可能带来的数值放大风险。
- 支持导出 CSV / JSON / Markdown / HTML 工程诊断报告。

## 一键安装

推荐 Python >= 3.10。

Linux / macOS：

```bash
bash install.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Windows CMD 或双击：

```bat
install.bat
```

安装脚本会自动创建 `.venv`、安装运行依赖、创建用户级全局启动器，并把启动器目录加入 PATH。安装完成后，新打开任意终端即可运行：

```bash
ckpt-viewer
```

安装脚本默认位置：

- Linux：启动器写入 `~/.local/bin/ckpt-viewer`
- Windows：启动器写入 `%USERPROFILE%\.local\bin\ckpt-viewer.cmd`
- 虚拟环境：项目内 `.venv`

可选环境变量：

- `CKPT_VIEWER_PYTHON=/path/to/python`
- `CKPT_VIEWER_VENV=/path/to/.venv`
- `CKPT_VIEWER_BIN_DIR=/path/to/bin`
- `CKPT_VIEWER_VENV_SYSTEM_SITE=0`

## 手动安装

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

## Web UI 使用

启动：

```bash
ckpt-viewer
```

或：

```bash
ckpt-viewer ui
```

默认访问：

```text
http://localhost:8501
```

## UI 效果
<img width="2523" height="990" alt="image" src="https://github.com/user-attachments/assets/29fbc1cc-2533-4c1c-8744-e75aa237c04a" />
<img width="2522" height="1256" alt="image" src="https://github.com/user-attachments/assets/c6d17204-5b28-4217-abf3-629fa547a320" />
<img width="2168" height="1149" alt="image" src="https://github.com/user-attachments/assets/e131ec3e-a888-405e-b96c-0ec7b7bd4105" />
<img width="2198" height="1199" alt="image" src="https://github.com/user-attachments/assets/91ed6bb4-8f14-4b43-bd37-a0b498b6c8d1" />
<img width="2185" height="1203" alt="image" src="https://github.com/user-attachments/assets/94d61776-0776-4dc5-a8f3-f20d0b0a9e35" />



## CLI 使用

分析单个模型权重：

```bash
ckpt-viewer analyze /path/to/model.pth --out report
```

导出完整报告：

```bash
ckpt-viewer analyze /path/to/model.pth --out report --export csv,json,md,html
```

排除 buffer，只关注可量化权重：

```bash
ckpt-viewer analyze /path/to/model.pth --out report --exclude-buffers --quantizable-only
```

按 role 过滤：

```bash
ckpt-viewer analyze /path/to/model.pth --filter-role weight,bias --out report
```

如果安全加载失败，并且你确认权重来自可信来源：

```bash
ckpt-viewer analyze /path/to/model.pth --trusted --out report
```

## 示例 checkpoint

仓库不附带二进制示例权重。需要测试安装时，可以生成一个很小的 synthetic checkpoint：

```bash
python examples/create_tiny_checkpoint.py
ckpt-viewer analyze examples/tiny_checkpoint.pth --out report --export csv,json,md,html
```

## 导出文件

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

## 安全说明

默认使用 PyTorch 安全加载：

```python
torch.load(path, map_location="cpu", weights_only=True)
```

只有确认 checkpoint 来自可信来源时，才使用 `--trusted` 或 Web UI 中的“信任该文件并使用 unsafe load”。不要对陌生来源的 `.pth` 文件启用 unsafe load。

## 限制

Ckpt_viewer 当前只做静态权重分析。它不分析 activation 分布，不执行模型 forward，不替代校准集和验证集评估。静态量化风险仅用于部署前预警，真实 PTQ / INT8 效果必须结合 calibration data、task metrics 和部署后 engine 验证。

# 

## License

MIT
