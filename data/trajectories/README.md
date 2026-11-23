# 共享上下文中的 user 消息重写

来源：[NVIDIA Nemotron-Agentic-v1](https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1)，`interactive_agent` 子集。作者 NVIDIA Corporation；许可 CC-BY-4.0；冻结 revision `650d590978ca35c8f1ecea2faf136e5fac421b62`。这是合成 agent 轨迹，不是真人日志。

本次下载该文件前 256 行。按结构规则获得 79 条候选：至少有两条 user 消息，并且目标 user 消息之前出现过工具返回。选择距离时间中点最近的合格 user 消息，每条轨迹最多一个切点。按轨迹 UUID 的固定 hash 排序，选择 16 条训练、4 条验证、4 条最终测试；不能把同一轨迹的不同片段分到两边。

保留切点之前的全部系统消息、用户消息、assistant 回复、历史工具调用和结果，以及完整工具定义。切点之后的内容不进入模型输入，也不作训练标签。两路接收者只有目标 user 消息不同，其他前缀与后缀 token 必须逐元素一致。发送者看到同样的历史、工具定义和当前消息，以通用编码任务提示词生成替换消息。

模型只生成该位置的下一条 assistant 回复或工具调用文本；不实际执行数据中的工具，不把原轨迹的未来工具结果强行接到新的分支。

资源上限在查看模型输出之前确定：sender/receiver 输入最多 6500 token；不截断历史。此次 256 行中有 177 行不满足中途 user + 历史工具返回要求，没有候选因输入长度被排除。明细见 `manifest.json`。

这是文件前部的小样本，主要来自食品配送客服环境，不能当作覆盖所有 agent 类型的实验。消息长度只衡量被重写的 user 内容，共享历史和固定模板单独记录。

验证集保留同历史下的空消息、无关消息对照。它们只用来检查状态距离对当前消息是否敏感，不加入 RL 奖励。试跑发现旧的长度权重可能让空消息占优，正式轮次将 lambda 调为 0.002；奖励仍只有末内容 token 状态余弦距离与消息长度两项。

复现准备：

```bash
python -m latent_protocol.prepare_trajectories
python -m latent_protocol.train --data-dir data/trajectories --contextual --output runs/trajectory-run --group-size 2 --message-tokens 96 --steps 16 --validation-items 4 --eval-every 8 --lr 0.0001 --length-weight 0.002 --final-test
```

原始下载缓存在 `raw/`（不纳入 git）；选中样本保留来源 UUID、切点索引、history、tools、context hash 和 tokenizer 长度。训练/测试划分独立于前一版手写短指令数据。
