# Xenosymbol：离散消息压缩实验

目标：训练发送者把指令 A 转成既有词表中的短消息 A'，使冻结接收者在完整生成之后的中间层表征接近处理 A 时的表征。

首轮真实运行已完成，结果、失败样本和限制见 [实验报告](reports/first_experiment.md)。在 8 条独立测试指令上，RL 将末状态距离从 0.170 降至 0.067；平均消息仍约 20 token。不能把这个结果称为已发现不可读的新语言。

现已增加用户提出的**完整轨迹模式**：保留系统策略、此前对话和工具结果，只替换中途一条 user 消息。[轨迹实验与具体例子](reports/trajectory_experiment.md)已经完成；这轮 16 步 RL 没有超过直接重写基线，负结果也完整保留。数据来源和运行命令见 [轨迹数据说明](data/trajectories/README.md)。

第一版固定目标：`reward = -(1 - cosine(h_last(A), h_last(A')) + lambda * len_tokens(A'))`。

比较口径：原文 A 是表征参照，A'=A 时距离按定义为 0。改写后距离较小只能称为偏差较小，不能称为相对于原文的距离改善。RL 是否有效，要在相同数据、编码提示和 lambda 下，比较发送者训练前产生的 A' 与训练后产生的 A'；不能把提示格式变化混入训练收益。

这里的“理解接近”是研究假设；实际测量是指定层向量的余弦距离。实验不把输出完全一致或任务正确率偷偷替换成训练目标。

## 实验约定

- 接收者：已有 Qwen 27B 的 NF4 量化底座，权重冻结。模型文件的真实 architecture 是 Qwen3_5ForConditionalGeneration，展示名称与架构名不同。
- 发送者：共用底座，只更新 q_proj/v_proj 上的 rank-8 LoRA；接收时禁用所有 LoRA。
- 表征：64 层中的第 32 层输出，层数从 1 开始计数。首个输出生成前和最后内容 token 被读入后的状态都保存。
- EOS/im_end 是传输结束符，不计入内容末 token 或消息长度。消息直接按 token ID 传输，不经过 decode/re-encode。
- 接收者贪心生成，关闭显式 thinking。完整生成后再做一次前向，显式读入末内容 token。生成达到上限时保存证据并报错，不能将截断状态当作完整输出。
- 发送者采样允许全部普通词表 token 和终止 token，禁止其他聊天/图像控制 token 和词表 padding 空位。
- RL：on-policy REINFORCE，独立采样候选，leave-one-out reward baseline。完整 action 序列的 log probability 求和，不按长度平均；包含实际采样的结束动作。采样和更新用相同温度和词表 mask。
- 首轮 lambda=0.005，学习率 3e-5；先跑试验，再基于验证集调整。其他相似度指标不进入奖励。
- 数据与系统提示词分离。24 个训练指令、8 个验证指令、8 个最终测试指令，仅作为工程与初步信号验证，不能支持通用新语言结论。

## 环境与运行

安装所需的版本和 GPU 自检命令见 [环境配置清单](environment/README.md)。项目根目录的 [requirements.txt](requirements.txt) 是安装入口；远端完整依赖快照和实测检查结果也已保存。

远端目录 `/workspace/latent-protocol`，Python `/opt/venv311/bin/python`。连接仅使用用户提供的交互 SSH + PTY；上传使用 shell 中的 base64 数据，不使用远端 exec/scp。

已实测远端 PyTorch 2.15.0+cu132、Transformers 5.6.0、PEFT 0.18.1、TRL 0.24.0。27B NF4 模型生成与 LoRA 反向传播峰值约 17.2GiB，GPU 为 RTX 6000 Ada 48GB。LLaMA Factory 已安装，其 PPO 路径主要接 value-head/reward API；本实验的奖励还要运行冻结接收者并抓中间层，因此使用独立的小型 RL 循环，便于核对精确奖励和状态索引。

```bash
python -m latent_protocol.make_data
python -m unittest discover -s latent_protocol/tests -v
python -m latent_protocol.train --output runs/pilot --steps 8 --eval-every 4 --validation-items 4
```

正式留出评估应在调参结束后加 `--final-test`；它比较本轮初始发送者和按验证集选择的最佳 checkpoint。若用 `--adapter` 继续训练，本轮 initial 是该输入 adapter，并不是未经训练的底座。每次输出目录必须新建，防止覆盖失败证据。

## 产物

- `config.json`：参数、真实模型配置、数据 hash、系统提示词、层位。
- `observations.jsonl`：A、A'、原始和压缩后的输出、长度、距离、奖励、截断/停止标记。
- `states/*.pt`：原始 token ID、生成前状态、末 token 状态、精确位置索引和完整文本。
- `events.jsonl`：逐步 RL、梯度范数、验证指标、运行时间、显存。
- `initial/`、`checkpoint-*/`、`best/`：发送者 LoRA checkpoint。
- `latest/`：每个成功更新后的最近 checkpoint，附优化器、随机状态、步数与训练顺序。`--adapter` 是 warm start，尚不是完整训练状态恢复入口。
- `summary.json`：验证最优点、接收者冻结不变量、可选的最终测试结果。

## 验收与解释

工程验收：奖励数值符合定义；同一个输入可复现状态；真实反向传播、非零有限梯度与 adapter 更新；训练后冻结接收者状态和文本不变；原始与压缩两侧的状态/文本可以回溯。

效果验收：在未参与优化的指令上，同时报告长度、表征距离和组合目标，相对于未训练发送者比较。不能把长度下降引起的总奖励改善说成表征距离也改善；不能把自然语言缩写说成已发现人类不可理解的新语言。

后处理对原始/压缩输出计算独立 embedding 相似度，与生成前和末 token 状态相似度计算相关性；另报字面输出相同率。相关性只用于分析，不参与 RL 和测试集 checkpoint 选择。所有失败轮次保留。

当前结果以实际生成的运行记录为准；启动训练不等于实验已完成。

运行审计和独立相关性分析：

```bash
python -m latent_protocol.audit runs/pilot
python -m latent_protocol.analyze runs/pilot --output runs/pilot-analysis
python scripts/plot_results.py runs/pilot --analysis runs/pilot-analysis
python scripts/summarize.py runs/pilot --output reports/pilot-summary.json
```

独立 embedding 使用 `intfloat/multilingual-e5-small`，双方输出均加 `query: ` 前缀，按 attention mask 平均池化并归一化。它只提供另一种测量尺度，不是人工语义等价评分。测试阶段只在调参结束后启用。

开发机现有 PyTorch 缺 CUDA 动态库，故本地只运行不依赖 torch 的测试；需要 torch 的测试在远端实际环境执行。不要将本地 skip 说成通过。
