# 环境配置清单

此前实验沿用远端已有的 `/opt/venv311` 环境。本目录记录实际运行的版本，提供安装步骤和自检，不需要构建镜像。

## 文件怎么用

| 文件 | 用途 |
|---|---|
| [../requirements.txt](../requirements.txt) | 本项目训练、分析、绘图的直接依赖，安装入口 |
| [observed/remote-freeze.txt](observed/remote-freeze.txt) | 远端完整版本快照，也作为传递依赖的版本约束 |
| [requirements-control.txt](requirements-control.txt) | 只在通过 SSH 控制远端的机器上安装 |
| [observed/runtime.json](observed/runtime.json) | 系统、GPU、Python、CUDA、包版本及 Python 源文件 RECORD 核查 |
| [observed/check.json](observed/check.json) | 远端 GPU/NF4 反向传播与模型目录自检结果 |
| [observed/resolution-summary.json](observed/resolution-summary.json) | 忽略已安装包、通过 PyPI 重新解析依赖的 dry-run 结果 |

完整快照包含服务器原先安装的 LLaMA Factory、TRL、Gradio 等。它们不都是本项目必需项；`requirements.txt` 只安装本项目所需的依赖及其依赖，不会因为作为约束就安装整份快照。

## 已验证的平台

| 项目 | 实测配置 |
|---|---|
| 系统 | Ubuntu 22.04.5 LTS，Linux x86_64 |
| Python | 3.11.18 |
| GPU | NVIDIA RTX 6000 Ada，48GB |
| 驱动 | 595.99.02（实测值，不声称是最低版本） |
| PyTorch | 包版本 2.15.0，实际导入版本 2.15.0+cu132 |
| PyTorch CUDA runtime | 13.2 |
| Transformers | 5.6.0 |
| PEFT / Accelerate | 0.18.1 / 1.11.0 |
| bitsandbytes | 0.50.2 |
| NumPy / SciPy / Matplotlib | 2.4.6 / 1.17.1 / 3.11.2 |

这份清单针对上述 Linux GPU 环境。CUDA runtime 随 Python 依赖安装，宿主机仍需可用的 NVIDIA 驱动。`nvidia-smi` 显示的 CUDA 版本不能代替 `torch.version.cuda` 的实际检查。

模型加载/短指令训练约占 17.4GiB，完整轨迹实验的已分配显存峰值约 23.3GiB；这些是测量值，不能直接当作其他机器的最低显存保证。

## 在 GPU 机器上安装

先准备 Python 3.11 和可用的 NVIDIA 驱动，然后在项目根目录执行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple pip==26.3
python -m pip install --index-url https://pypi.org/simple -r requirements.txt
python -m pip check
python scripts/check_environment.py
```

本次核对的 PyPI torch 2.15.0 的 Linux 依赖指定 CUDA toolkit 13.2.2，实际环境导入为 cu132。安装后自检会核对版本、GPU/BF16 可用性，并真正运行一个小型 NF4 层的前向和反向传播。它不会加载 27B 权重。

直接依赖和传递依赖按已用版本约束，不需要手工逐个配置 CUDA 动态库。若出现 `libcudart` / `libcublasLt` 缺失，先检查是否在正确虚拟环境、安装是否完整；不能仅因 `importlib` 能找到 torch 就认定环境正常。

## 模型文件独立准备

实验使用的现有目录是 `/workspace/model_nf4_native`，真实架构为 `Qwen3_5ForConditionalGeneration`，预量化方式为 bitsandbytes NF4。共有 26 个 safetensors 分片，需连同 config、权重索引、tokenizer 和聊天模板等文件一起保留。模型展示名称不等于架构名称。

权重不在 Git 中，requirements 也不会下载它。将同一份预量化模型放到自己的路径后：

```bash
python scripts/check_environment.py --model /path/to/model_nf4_native
python -m unittest discover -s latent_protocol/tests -v
python -m latent_protocol.train \
  --model /path/to/model_nf4_native \
  --output runs/environment-smoke \
  --steps 1 --group-size 2 --eval-every 1 --validation-items 1
```

模型自检只检查目录、分片存在性和聊天模板，不代替完整训练验证。上面最后一条才会实际加载模型并进行一次 RL 更新，输出目录必须尚不存在。模型元数据与分片大小记录见 [reports/environment.json](../reports/environment.json)。

模型代码使用现成的 PyTorch/Transformers 路径，不要求额外安装 FlashAttention、flash-linear-attention 或 causal-conv1d。此前运行日志里的 fast-path 提示意味着使用了较慢的实现，不表示实验没有运行。

## 只在本机控制远端

如果本机只负责上传、监控、下载，不需要安装 GPU 依赖：

```bash
python -m pip install -r environment/requirements-control.txt
command -v ssh
command -v socat
python scripts/remote.py run 'nvidia-smi'
```

`ssh` 和 `socat` 是系统程序，Ubuntu/Debian 通常由 `openssh-client`、`socat` 包提供。连接地址、代理和密钥路径在 `scripts/remote.py` 中；密钥本身不在仓库。该工具严格使用交互 PTY，没有 SSH exec/scp 通道。

## 核验范围与更新清单

已经完成：

- 现有远端环境 `pip check` 通过。
- 核心包列在 wheel RECORD 中的 Python 文件未发现本地修改；这不是所有二进制文件的完整供应链审计。
- GPU 上真实 NF4 前向/反向传播通过，Qwen3.5 类和 PEFT 导入成功。
- 现有模型的 26 个分片和聊天模板检查通过。
- 使用 `--dry-run --ignore-installed` 成功解析本项目依赖，避免仅靠“现有包已满足”掩盖安装冲突。

没有把 dry-run 称为“已在全新机器完整安装并重训”。原来的真实训练记录与本次环境检查是两份不同证据。清单是已观察环境的版本快照，不是带 wheel 文件的离线安装包。

需要再次导出时，在目标训练环境运行：

```bash
python scripts/export_environment.py --output /tmp/xenosymbol-environment
python scripts/check_environment.py --model /path/to/model_nf4_native \
  --output /tmp/xenosymbol-environment/check.json
```

导出程序遇到 URL/本地路径形式的依赖会要求人工查看来源，不会把它们直接作为可公开的版本清单写出。
