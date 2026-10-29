# SP（Sequence Parallel）实现说明

> 配套文档：[parallel_state.md](parallel_state.md) 讲通信组与通信原语；本文讲 SP 在模型层落地时改了哪几处、为什么必须改在这几处，以及有卡实测结论。
>
> 代码版本：`25d0b1c`。

## 一、SP 要做什么

把 prefill 阶段激活的 token 维从 T 切成 T/N，且只在"残差流"区段生效。

关键点：SP 不是给每个算子都加一次切分，而是**只在 token 维进出 shard 区的边界上做通信**，区段内部算子零改动。

为什么恰好是四个边界点：Megatron 式 TP 的层内结构是 Column → (激活) → Row。从 token 维看，embedding 是"完整 T 的入口"，lm_head 是"完整 T 的出口"，每个 Column 层是"需要完整 T"的边界，每个 Row 层是"产出完整 T"的边界。

    token-sharded（T/N 行）
        │ all_gather(total_len=T)      ← Column 入口：补齐 token 行
        ▼
    完整 T 行 → qkv_proj → RoPE → attention → o_proj
        │ reduce_scatter              ← Row 出口：求和后只留自己那份
        ▼
    token-sharded → RMSNorm → MLP 同样模式

四个点靠同一个约定串联：[parallel_state.token_range](parallel_state.py) 是所有 rank 用同一公式、对同一个 T 各自算出的连续区间，切和拼天然对齐，不需要通信交换元数据；`num_tokens` 由 prefill 的 `set_context` 从 `cu_seqlens_q[-1]` 取一次。

## 二、四个接管点

| 接管点 | 位置 | 方向 | 通信原语 |
| --- | --- | --- | --- |
| embedding 出口 | [embed_head.py L44-46](../layers/embed_head.py) | 全量 T → shard | `token_range` 本地切片 |
| Column 入口 | [linear.py L75-76](../layers/linear.py) | shard → 全量 T | `all_gather` |
| Row 出口 | [linear.py L157-158](../layers/linear.py) | 全量 T → shard | `reduce_scatter` |
| lm_head 入口 | [embed_head.py L64-66](../layers/embed_head.py) | shard → 全量 T | `all_gather` |

### 1. embedding 出口：全量算完再切

```python
def forward(self, x: torch.Tensor):
    context = get_context()
    if self.tp_size > 1:
        mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
        x = mask * (x - self.vocab_start_idx)
    y = F.embedding(x, self.weight)
    if self.tp_size > 1:
        y = mask.unsqueeze(1) * y
        # 词表维归约按行号对齐，各 rank 必须持同一批 token，所以这里先算全量 T
        get_tp_group().all_reduce(y)
    if context.enable_sp and context.is_prefill:
        start, end = get_tp_group().token_range(context.num_tokens)
        y = y[start:end].clone()
    return y
```

为什么不能"先切 `input_ids` 再算"：

- 这里的 all_reduce 是**词表维**的。每个 rank 只持有 vocab/N 行权重，靠 mask 让不属于自己词表段的 token 输出 0，再把各 rank 的同一行相加。
- 它要求**各 rank 的第 i 行是同一个 token**。先切的话，rank0 第 i 行是 token i、rank1 第 i 行是 token T/N+i，相加得到两个无关 token 的向量和。
- 症状很隐蔽：两个 rank 拿到的是**同一份**错误结果，模型照常出 logits，表现为"所有序列输出一模一样"。
- 代价：入口是"全量计算 + 全量归约"≈2V（V = T×hidden），比 Megatron 的 reduce_scatter_embeddings 贵，v2 换原生 reduce_scatter 再降。

出口的 `.clone()` 不能省：`y[start:end]` 是 view，底层仍引用 T×H 存储，父 tensor 释放不掉，激活显存收益归零。

**我个人理解**：本来就是原来的 all_reduce 换 reduce_scatter，ai 自己奇思妙想说先切再算。先切再算切出来对不上词表 id 的位置了：每个 rank 都有自己输出的对应 id 的，你先切了，没有那么多 0 补齐，每个 rank 的 id 就对不上了。

举例（token id = 6 落在 rank1 的词表段 4~7）：

```
  未切：rank0 行 = 0，rank1 行 = embed(6)  → all_reduce → 完整
  先切：rank0 留着这个 token 但算出 0；rank1 已把它切掉、手里没有
        → all_reduce 后这行永远是 0
```

### 2. Column 入口：gather 回完整 T

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ctx = get_context()
    if ctx.enable_sp and ctx.is_prefill:
        x = get_tp_group().all_gather(x, ctx.num_tokens)
    return F.linear(x, self.weight, self.bias)
```

为什么：

- Column 权重沿**输出维**切，输入维是完整 hidden。此时激活已经是 T/N 行 × 完整 hidden——hidden 不缺，缺的只是 token 行，所以沿 dim0 gather。
- 不能用 all_reduce：各 rank 持有的是**不同 token**，不是同一 token 的部分和，求和没有意义。
- 一处改动覆盖全部：`MergedColumnParallelLinear` / `QKVParallelLinear` 只重写 `weight_loader`，forward 直接继承这里。

### 3. Row 出口：求和后只留自己那份

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
    if self.tp_size > 1:
        ctx = get_context()
        if ctx.enable_sp and ctx.is_prefill:
            y = get_tp_group().reduce_scatter(y)
        else:
            get_tp_group().all_reduce(y)
    return y
```

为什么这里可以切、而 embedding 不行：Row 权重沿**输入维**切，输出是完整 hidden 的**部分和**。各 rank 的第 i 行是同一个 token 的不同 hidden 片段，逐行相加正好补齐——行是对齐的。求和后再按 `token_range` 只留自己那段，正是 reduce_scatter 的定义。

bias 仍然只在 `tp_rank == 0` 加，`reduce_scatter` 内部先 all_reduce，语义不变。

### 4. lm_head 入口：先恢复完整 T 再选 last token

```python
if context.is_prefill:
    if context.enable_sp:
        # last_indices 是全局坐标系，必须先恢复完整 T
        x = get_tp_group().all_gather(x, context.num_tokens)
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

为什么顺序不能反：`cu_seqlens_q[1:] - 1` 是**全局坐标**，SP 下 x 只有 T/N 行，拿全局下标索引本地张量会越界或取错行。

由此可提炼一条通用规律：**坐标系类**索引（`cu_seqlens` / `last_indices` / `positions`）必须与张量同坐标系，通信要排在它前面；**地址类**索引（`slot_mapping` / `block_tables`）的条目跟着同区间切即可，不需要提前 gather。

## 三、什么算子能留在 shard 区

判据三条，缺一不可：

1. 逐 token 独立（不做跨 token 归约，例如 attention 的 softmax 只在同一条序列内做，可以；跨序列做就不行）；
2. 元数据可同区间切片（`cu_seqlens`、`positions` 这类要能跟着切）；
3. 形状与行顺序不变。

可用的等价式自检：`f(x_full)[s:e] == f(x_full[s:e], meta[s:e])`。

按这三条判据，残差 add、RMSNorm、激活函数天然满足；RoPE、attention 满足但成本收益比差（内部中间量仍是完整 T），所以保持完整 T 区不动，只切残差流。

## 四、开关链路

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| config.py L22 / L41-47 | `enable_sp` 字段 + 断言 | tp>1 才有切分空间；首版限 eager；MoE 专家循环会把 Column gather 的结果在 Row 处放大 N 倍，多模态走 `inputs_embeds` 绕过本文二.1，两者必须拦死 |
| arg_utils.py L17 / L51 | `--enable-sp` / `-sp` | argparse 自动映射 `dest=enable_sp` |
| arg_utils.py L30-31 | `__post_init__` 中 `enable_sp` 时强制 `enforce_eager=True` | `--enforce-eager` 是 `store_true` 且无 default，argparse 会给 False 覆盖 dataclass 的 True |
| context.py L8-9 / L29-47 | `enable_sp` / `num_tokens` 字段 | `Context(...)` 必须**关键字构造**：插字段会让位置参数整体错位且静默错算 |
| model_runner.py L309-319 | prefill 传 `num_tokens=int(cu_seqlens_q[-1].item())` | `item()` 是 D2H 同步，每层算一次会拖慢；`total_len` 全网一致，只算一次 |
| model_runner.py decode 两处 | 不传，走默认 False | 双保险：即使调用方忘了，decode 也不会进 SP 分支 |
| api_server.py L81/L109、bench.py L27 | 透传 | 原代码读 `args.enable_sp` 却没定义 parser，属静默失效 |

另外：SP **不新建进程组**，直接复用 `get_tp_group()`。SP 的 rank 集合与 TP 组完全一致（同一批卡上叠加），新建组没有通信收益，只增加建组握手成本。

## 五、有卡实测（Ascend 910B ×2，Qwen3-0.6B，TP=2）

条件：`enforce_eager=True`、`max_num_batched_tokens=16384`、`max_model_len=4096`、`gpu_memory_utilization=0.5`；计时为单次 prefill step 的 3 次中位数；显存口径为 `reset_peak_memory_stats()` 后单次 prefill 的 `allocated_bytes.all.peak − 调用前 current`。

正确性（prefill last-token logits 跨进程比对；采样序列在 decode 阶段跨进程会分叉，基线自己跑两次也会，不能作判据）：

| case | 总 token | max\|Δ\| 基线 vs SP | top1 |
| --- | --- | --- | --- |
| 3+3+1 | 7 | 0 | ✅ |
| 5+5+3 | 13 | 0 | ✅ |
| 256+256 | 512 | 0 | ✅ |
| 单条 | 4096 | 0 | ✅ |

覆盖整除与不整除，均逐位一致。通信原语在 gloo 下 `(tp, cp) = (2,1)/(2,2)/(4,2)`、T=7/8/13 全过。

embedding 分片语义（合成权重，词表 8 / hidden 4，第 v 行填 v）：修复前两个 rank 都得到 `[4,6,8,10]`——第 j 行是"token j 在 rank0 词表段的部分"加"token T/2+j 在 rank1 词表段的部分"，即二.1 描述的错法；修复后 rank0 `[0,1,2,3]`、rank1 `[4,5,6,7]`，与"单卡全量 embedding 的对应行"一致。这类错误形状完全正常，只有比对数值才抓得到。

性能：

| 总 token | 基线 prefill | SP prefill | 时延 | 激活峰值 基线 → SP | KV blocks |
| --- | --- | --- | --- | --- | --- |
| 2048 | 56.97 ms | 71.90 ms | +26.2% | 145.0 → 137.0 MB（−5.5%） | 2148 / 2148 |
| 8192 | 89.57 ms | 94.83 ms | +5.9% | 252.2 → 220.2 MB（−12.7%） | 2148 / 2147 |
| 16384 | 173.47 ms | 183.52 ms | +5.8% | 472.3 → 408.3 MB（−13.5%） | 2148 / 2148 |

结论：

a. **正确性达标**，但显存收益很薄。绝对量约每 1k token 4 MB（对应两个 T×H 张量），落到整机 30.8 GB 上只有 0.03~0.2%。

b. **换不成容量**。`allocate_kv_cache` 在权重加载完、尚未跑任何 forward 时就按当时的 peak（≈权重占用）算死 `num_kvcache_blocks`，SP 省的激活那时还不存在，实测块数未变。所以不要用"KV 块是否变多"判断 SP 是否有效；唯一变现路径是"同 T 峰值更低 → 可以开更大的 `max_num_batched_tokens` 而不 OOM"。

c. **时延净亏**。8K/16K 档 +5.8%，2K 档 +26%（固定通信延迟摊不薄）。

d. **+5.8% 里约 1/3 是实现成本**。按环形带宽折算，基线每层 4 单位，SP 每层 6 单位；其中 2 单位来自"伪 reduce_scatter"（all_reduce + 切片，语义对但通信量翻倍）。换原生 `reduce_scatter_tensor` 可回到 4 单位，但需要把 `token_range` 与 `all_gather` 的分片规则一起改成等长块语义（原生只支持等长块，余数会堆到尾部 rank，`T < tp` 时还会出现 0 行 chunk），属方案级改动。

e. **当前配置下净收益≈0**。这符合 SP 的定位（长序列 prefill、激活占比高）：要转正需要更大模型、32K+ 长 T，或压低 KV 分配让激活占比变高。

## 六、遗留与后续

遗留：

- `T < tp_size` 时 `token_range` 会产生 0 行区间（如 T=3、TP=8 → `sizes=[1,1,1,0,...]`），`all_gather` 会对空 tensor 发起集合通信，HCCL 下行为未验证。建议加兜底或补用例。
- `attention.py` prefill 分支的 q/k/v 在 CANN 9.1 下需要 `.contiguous()`（`FusedInferAttentionScore` 报 `EZ0037`），基线与 SP 都受影响，尚未提交。
- 未覆盖：32K 长序列、TP=4/8、profiler 通信占比拆解。

v2 方向（按收益排序）：

1. 原生 `reduce_scatter_tensor` + 等长块语义：Row 出口与 embedding 入口都从 2V 降到 ≈V，是当前时延亏损的主要来源。
2. 异步化：`async_op=True` + `work.wait()`，把 gather 塞进 RMSNorm 的计算间隙。
3. MoE / 多模态支持：前者要把专家循环改成 block 级 SP，后者要接管 `inputs_embeds` 入口。
