# SP（Sequence Parallel）实现说明

> 配套文档：[parallel_state.md](parallel_state.md) 讲通信组与通信原语；本文讲 SP 在模型层落地时改了哪几处、为什么必须改在这几处，以及有卡实测结论。
>
> 代码版本：`25d0b1c`。

## 〇、符号与术语

| 符号 | 含义 |
| --- | --- |
| `T` | 一个 prefill step 里打包后的**总 token 数**，即 `cu_seqlens_q[-1]`。多条请求拼在一起，所以 T 可能不整除 N |
| `N` | `tensor_parallel_size`，即 TP 组大小 |
| `V` | 一次通信的数据量，`V = T × hidden_size`。"2V""6 单位"里的单位就是 V（按环形带宽折算，`all_reduce ≈ 2V`、`all_gather ≈ V`、原生 `reduce_scatter ≈ V`，均取 N 较大时的近似） |
| full T 区 | 激活持有全部 T 行的区段。attention、以及每对 Column→Row 的整段都在这里跑（原因见 1.2） |
| shard 区 | 激活只有 T/N 行的区段。SP 想扩展的就是这一段 |
| `token_range(T)` | [parallel_state.py](parallel_state.py) 里的方法。所有 rank 用**同一个公式**对自己的 rank 算出自己负责的连续行区间 `[start, end)`，余数分给靠前的 rank。它不需要任何通信，因为每个 rank 都能独立算出全部区间的划分 |
| `num_tokens` | 全局 T，在 prefill 的 `set_context` 里算一次存进 `Context`，避免每层重复做 D2H 同步 |

## 一、SP 要做什么

把 prefill 阶段激活的 token 维从 T 切成 T/N，且只在"残差流"区段生效。

关键点：SP 不是给每个算子都加一次切分，而是**只在 token 维进出 shard 区的边界上做通信**，区段内部算子零改动。

### 1.1 四个接管点在模型中的位置

先看模型长什么样。以 Llama 为例（其他 decoder-only 模型结构相同），`LlamaForCausalLM` 的结构是：

```
LlamaForCausalLM
├─ model.embed_tokens                      ① 出口切片  full T → shard
├─ model.layers × num_hidden_layers
│   ├─ input_layernorm   (= RMSNorm，内含残差 add)      shard
│   ├─ self_attn.qkv_proj                 ② 入口 gather  shard → full T
│   ├─ RoPE + attention                               full T
│   ├─ self_attn.o_proj                   ③ 出口 scatter full T → shard
│   ├─ post_attention_layernorm (= RMSNorm，内含残差 add) shard
│   ├─ mlp.gate_up_proj                   ② 入口 gather  shard → full T
│   ├─ mlp.act_fn (SiLU·Mul)                          full T
│   └─ mlp.down_proj                      ③ 出口 scatter full T → shard
├─ model.norm   (= RMSNorm，内含残差 add)              shard
└─ lm_head                                ④ 入口 gather  shard → full T
```

对应代码：[llama.py L192-210](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/models/llama.py#L192-L210) 是 `LlamaModel.forward` 的层级，[llama.py L167-182](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/models/llama.py#L167-L182) 是单层的调用顺序，[llama.py L241](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/models/llama.py#L241) 是 `compute_logits`。

从这张图能直接读出三件事：

1. **①和④是模型两端的出入口**。`embed_tokens` 吃 `input_ids`、吐 hidden；`lm_head` 吃 hidden、吐 logits。SP 的切分区间就是"①之后 → ④之前"，这两处必然要做形态转换。
2. **②③在每层各出现两次**（attention 一对、MLP 一对）。所以"四个接管点"不是四段并列代码，而是**2 个端点 + 2 类每层复现的边界**。因为改的是 `ColumnParallelLinear` / `RowParallelLinear` 基类的 `forward`，一处改动自动覆盖 `qkv_proj`、`gate_up_proj`、`o_proj`、`down_proj` 以及所有层。
3. **shard 区里只剩 RMSNorm**（这个实现里残差 add 融合进了 `NPURMSNorm` 的第二个返回值）。这正是判据能成立的原因，见第三节。

### 1.2 为什么边界恰好落在这两类层上

Megatron 式 TP 的层内结构固定是 `Column → (激活) → Row`。这里要注意一个容易搞混的点：**从算子形状看，Token 维几乎处处可切**——`Column`、激活函数、甚至 `Row` 的 matmul 本身都是逐 token 的，来几行算几行。所以"边界在哪"不是被算子形状决定的，而是被下面这条硬约束决定的：

- **`Row` 的跨 rank 归约必须行对齐**。Row 沿输入维切，各 rank 算出的都是**部分和**，要靠一次跨 rank 求和补齐；这次求和是按行号逐元素加的，要求各 rank 的第 i 行是同一个 token。
- 而 SP 的 shard 恰恰是"**每个 rank 持不同的行**"（rank r 持 `[s_r, e_r)`）。所以只要张量处于 shard 状态，Row 的归约就会把不同 token 加在一起。
- 结论：**进入一对 Column→Row 之前必须已恢复 full T，离开 Row 之后才能重新切走**。恢复得越早，gather 的数据越便宜——`Column` 入口处张量宽度是 hidden（H），若拖到 `Row` 入口才 gather，宽度是中间维（`2I/N`，通常比 H 大），所以选在 Column 入口 gather。

于是边界就落在"每一对 Column 的入口"和"每一对 Row 的出口"上：

```
shard（T/N 行）
    │ all_gather(ctx.num_tokens)     ← ② Column 入口
    ▼
full T 行 → qkv_proj → RoPE → attention → o_proj
    │ reduce_scatter                 ← ③ Row 出口
    ▼
shard → RMSNorm → 下一个 Column 入口
```

## 二、四个接管点

| 接管点 | 模型中的位置 | 代码位置 | 方向 | 原语 |
| --- | --- | --- | --- | --- |
| ① embedding 出口 | `embed_tokens`，模型入口 | [embed_head.py L44-46](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/layers/embed_head.py#L44-L46) | full T → shard | `token_range` 本地切片 |
| ② Column 入口 | `qkv_proj` / `gate_up_proj`，每层两次 | [linear.py L75-76](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/layers/linear.py#L75-L76) | shard → full T | `all_gather` |
| ③ Row 出口 | `o_proj` / `down_proj`，每层两次 | [linear.py L157-158](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/layers/linear.py#L157-L158) | full T → shard | `reduce_scatter` |
| ④ lm_head 入口 | `lm_head`，模型出口 | [embed_head.py L64-66](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/layers/embed_head.py#L64-L66) | shard → full T | `all_gather` |

### 2.1 ① embedding 出口：全量算完再切

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

这一处为什么不能"先切 `input_ids` 再算"：

- 这里的 `all_reduce` 归约的是**词表维**。每个 rank 只持有 `vocab/N` 行权重，靠 `mask` 让不属于自己词表段的 token 输出 0，再把各 rank 的同一行相加，才能拼出一个 token 的完整 embedding 向量。
- 它要求**各 rank 的第 i 行是同一个 token**（行号对齐）。先切的话，rank0 第 i 行是 token `i`，rank1 第 i 行是它自己那段里的另一个 token（如 T=8、N=2 时是 token `4+i`），相加得到两个无关 token 的向量和。
- 更糟的是会整行变零：一个 token 的 id 只落在某一个 rank 的词表段里，其余 rank 上那一行本来就是 0、要靠对方补齐；而对方把这个 token 切掉了，于是这行永远是 0。
- 所以"先切"既没省通信（词表维是分的，`all_reduce` 躲不掉），又破坏了行对齐和 token 覆盖，是纯负收益。正确的做法只有"全量算 + 全量归约 + 出口留自己那段"，也就是原生 `reduce_scatter` 的语义。

设定举例（`vocab=8`、`N=2`，rank0 持有 id `0~3`、rank1 持有 `4~7`，token id = 6）：

```
  未切：rank0 该行 = 0，rank1 该行 = embed(6)   → all_reduce → 完整
  先切：rank0 留着这个 token 但只能算出 0；rank1 已把它切掉、手里没有
        → all_reduce 后这行永远是 0
```

出口的 `.clone()` 不能省：`y[start:end]` 是 view，底层仍引用整个 T×H 存储，父 tensor 释放不掉，激活显存收益归零。

> **实现记录（复盘）**：本来就是原来的 all_reduce 换 reduce_scatter，ai 自己奇思妙想说先切再算。先切再算切出来对不上词表 id 的位置了：每个 rank 都有自己输出的对应 id 的，你先切了，没有那么多 0 补齐，每个 rank 的 id 就对不上了。

代价：入口要多做一次**全量 T** 的 embedding 计算和一次全量归约（≈ 2V），比 Megatron 的 `reduce_scatter_embeddings`（边算边只留自己那段，≈ V）贵，v2 换原生 `reduce_scatter` 可以再降。

### 2.2 ② Column 入口：gather 回完整 T

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ctx = get_context()
    if ctx.enable_sp and ctx.is_prefill:
        x = get_tp_group().all_gather(x, ctx.num_tokens)
    return F.linear(x, self.weight, self.bias)
```

为什么：

- Column 本身**对行数没有要求**（逐 token 的 matmul，来几行算几行），所以这里 gather 不是为了满足 Column，而是为了满足紧随其后的 Row 的行对齐要求（见 1.2）。Column 沿输出维切、输入维是完整 hidden，所以激活只缺 token 行、不缺 hidden 片段，沿 dim0 gather 恰好补上。
- 不能用 `all_reduce` 代替：此刻各 rank 持有的是**不同 token**，不是同一 token 的部分和，求和没有意义。
- 一处改动覆盖全部：`MergedColumnParallelLinear` / `QKVParallelLinear` 只重写 `weight_loader`，`forward` 直接继承这里。
- gather 写在 `forward` 内部而不是在 `RMSNorm` 之后单独一句，是因为 shard 区里所有算子（含 RMSNorm）都不需要改，只有真正需要 full T 的区段才在自己入口补齐。

### 2.3 ③ Row 出口：求和后只留自己那份

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

为什么这里可以切、而 embedding 不行：Row 权重沿**输入维**切，输出是完整 hidden 的**部分和**。各 rank 的第 i 行是同一个 token 的不同 hidden 片段，逐行相加正好补齐——**行是对齐的**。求和后再按 `token_range` 只留自己那段，正是 `reduce_scatter` 的定义。embedding 那边的各 rank 第 i 行不是同一个 token，所以做不了同样的处理。

`bias` 只在 `tp_rank == 0` 加，原因是 bias 是每个 rank 都能算出的完整向量，若每个 rank 都加，`all_reduce` 后会被累加 N 次；只在 rank0 加，归约后正好一次。

注意当前 `reduce_scatter` 是**模拟实现**：内部先 `all_reduce`（2V）再本地切片，语义正确但通信量是原生实现（V）的两倍，见 [parallel_state.py L34-41](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/utils/parallel_state.py#L34-L41)。

### 2.4 ④ lm_head 入口：先恢复完整 T 再选 last token

```python
if context.is_prefill:
    if context.enable_sp:
        # last_indices 是全局坐标系，必须先恢复完整 T
        x = get_tp_group().all_gather(x, context.num_tokens)
    last_indices = context.cu_seqlens_q[1:] - 1
    x = x[last_indices].contiguous()
```

两步各自的理由：

1. **为什么只取一部分行**：prefill 阶段每条请求只需要**最后一个 token** 的 logits（后续 decode 由采样结果继续），而多条请求被拼成了一维 T，所以 `cu_seqlens_q[1:] - 1` 就是"每条序列最后一个 token 的全局行号"。
2. **为什么 gather 必须在取行之前**：这串行号是**全局坐标系**下的下标，而 SP 下 `x` 只有 T/N 行，拿全局下标索引本地张量会越界或取错行。所以顺序只能是"先恢复完整 T，再按全局下标挑行"。

由此提炼一条通用规律：

| 索引类型 | 例子 | 通信要求 |
| --- | --- | --- |
| **坐标系类** | `cu_seqlens_q`、`last_indices`、`positions` | 必须与张量同坐标系。张量被切了就必须先通信恢复，通信排在索引操作**之前** |
| **地址类** | `slot_mapping`、`block_tables` | 每个条目独立对应一个 token 的落盘位置，跟着同区间切即可，不需要提前 gather |

## 三、什么算子能留在 shard 区

判据三条，缺一不可：

1. **逐 token 独立**：输出第 i 行只依赖输入第 i 行，不做跨 token 归约。
2. **元数据可同区间切片**：算子若依赖跨 token 的元数据（`cu_seqlens` 等），这些元数据必须能按同一区间切。
3. **形状与行顺序不变**：输出仍是 `(T/N, hidden)`，且行序与输入一致。

自检用的等价式：

```
f(x_full)[s:e]  ==  f(x_full[s:e], meta[s:e])
   全量算再切         切成 shard 后直接算
```

其中 `[s:e]` 就是 `token_range(T)` 给出的本 rank 区间，`meta` 是判据 2 里的元数据。两边相等，说明这个算子可以安全地留在 shard 区。

注意：**满足判据只说明"可以切"，不等于"实际切了"**。边界放在哪里还会影响成本，见 1.2 里"越早 gather 越便宜"的取舍。

按这三条判据逐个看：

- **残差 add、RMSNorm、SiLU·Mul：可以留**。三者都是逐元素、逐 token 运算，满足全部三条。这个实现把残差 add 融合进了 `NPURMSNorm`，所以 shard 区里只剩归一化这一种算子；`SiLU·Mul` 虽然也满足判据，但当前边界选在 `gate_up_proj` 入口，它实际跑在 full T 区。
- **attention：不能留**。因果注意力里第 i 个 token 的输出要看到前面所有 token 的 K/V，不满足判据 1；它依赖的 `cu_seqlens_q` 是全局前缀和，切成局部区间就失去语义，也不满足判据 2。所以 attention 只能在 full T 区跑。
- **RoPE：理论上能留，但不划算**。它只依赖 token 自身和位置编码，满足三条判据；但它前后紧挨着 attention（full T）和 `qkv_proj`/`o_proj`，单独切它会在两端各多出一次 gather/scatter，净亏。所以随 attention 一起放在 full T 区。

结论：**只切残差流**（`RMSNorm` 这一小段），attention 与 MLP 的 matmul 全部保持 full T。这也是 SP 收益受限于"激活中残差流占比"的原因。

## 四、开关链路

| 文件 | 改动 | 为什么 |
| --- | --- | --- |
| [config.py L22](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/config.py#L22-L22) | `enable_sp: bool = False` | 配置开关 |
| [config.py L41-47](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/config.py#L41-L47) | 4 条断言 | 见下方"四条断言" |
| [arg_utils.py L17](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/utils/arg_utils.py#L17-L17) / [L51-52](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/utils/arg_utils.py#L51-L52) | `--enable-sp` / `-sp` | argparse 自动映射 `dest=enable_sp` |
| [arg_utils.py L26-31](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/utils/arg_utils.py#L26-L31) | `__post_init__` 里 `enable_sp` 时强制 `enforce_eager=True` | `--enforce-eager` 是 `store_true` 且没有 default，argparse 会给 False，把 dataclass 默认的 True 覆盖掉 |
| [context.py L8-9](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/utils/context.py#L8-L9) / [L29-47](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/utils/context.py#L29-L47) | 新增 `enable_sp`、`num_tokens` 字段 | `Context(...)` 必须用**关键字构造**：插字段会让位置参数整体错位，而且不会报错、只会算错 |
| [model_runner.py L309-319](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/engine/model_runner.py#L309-L319) | prefill 传 `num_tokens=int(cu_seqlens_q[-1].item())` | `item()` 是 D2H 同步；每层算一次会拖慢整个 forward，所以在一个 step 开头算一次存进 Context |
| model_runner.py decode 两处 | 不传，走默认 `enable_sp=False` | 双保险：即使调用方忘了，decode 也不会进 SP 分支 |
| [api_server.py L81](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/v1/entrypoints/openai/api_server.py#L81-L81) / [L109](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/v1/entrypoints/openai/api_server.py#L109-L109)、[bench.py L27](file:///home/d00949339/workspace/nano-vllm-ascend/bench/bench.py#L27-L27) | 透传 `enable_sp` | 原代码读 `args.enable_sp` 但 parser 里没定义，属静默失效 |

**四条断言**（都在 `hf_config` 加载之后，否则拿不到 `architectures` / `model_type`）：

- `tensor_parallel_size > 1`：N=1 时 T/N = T，没有切分空间。
- `enforce_eager == True`：首版只在 eager 下验证过，图模式未适配 dist 算子。
- 禁止 MoE：MoE 的专家循环里每个专家都是一对 Column/Row，若在 block 级复用同一套 gather/scatter，`Column` gather 出来的结果会在 `Row` 处被按专家数重复归约，等于放大 N 倍。
- 禁止多模态：`qwen3_vl` / `qwen2_5_omni` 走 `inputs_embeds` + `masked_scatter` 绕过 `embed_tokens`，即绕过了本文 2.1 的 ①，切分区间失去起点。

另外：**SP 不新建进程组**，直接复用 `get_tp_group()`。SP 的 rank 集合与 TP 组完全一致（同一批卡上叠加），新建组没有通信收益，只多一次建组握手。

## 五、有卡实测（Ascend 910B ×2，Qwen3-0.6B，TP=2）

环境：`vllm-qxy` 容器（Python 3.12 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 / CANN 9.1），Ascend 910B ×2；模型 `/mnt/weight/Qwen3-0.6B`（28 层，H=1024，16 头 / 8 KV 头，I=3072，vocab 151936）。

配置与口径：`enforce_eager=True`、`max_num_batched_tokens=16384`、`max_model_len=4096`、`kvcache_block_size=256`、`gpu_memory_utilization=0.5`（整机总显存 30.8 GB）；token ids 固定，同形状预热 1 次，重复 3 次取中位数，计时边界为 `llm.step()` + `npu.synchronize()`。

另有一处本地改动未提交：`attention.py` prefill 分支对 `q/k/v` 加 `.contiguous()`。CANN 9.1 下 `FusedInferAttentionScore` 在非 PA 场景强制要求 value 连续，缺失会报 `EZ0037`，基线与 SP 都跑不起来，与本特性无关（见第六节）。

### 5.1 验证设计

分四层，每层解决上一层定位不到的问题：

| 层 | 怎么验 | 抓什么 | 为什么不能只做这层 |
| --- | --- | --- | --- |
| ① 通信原语 | gloo + CPU 多进程，直接调 `token_range` / `all_gather` / `reduce_scatter` | 区间划分、pad 与截断、余数分配 | 只证明原语对，不证明模型用对了 |
| ② 模块语义 | embedding 用合成权重（词表 8、hidden 4、第 v 行填 v），比 SP 输出与"单卡全量 embedding 对应行" | **行错位**这类形状正常、数值全错的 bug | 需要人工构造可辨认的权重，端到端跑真实模型看不出来 |
| ③ 端到端数值 | 真实模型 prefill，基线 vs SP 跨进程比 last-token logits | 各接管点联动后的整体正确性 | 定位不到是哪一处错 |
| ④ 性能/显存 | 同输入、同权重、只改 `enable_sp` 的单变量对照 | SP 的实际收益与成本 | 正确性达标不代表有价值 |

几条设计上的取舍：

- **先验基线自身可复现**。同一份输入让基线跑两次，`max|Δ|` 必须是 0；否则后面"基线 vs SP"的差值分不清是 SP 引入的还是环境本身不确定。四个 case 基线自比均为 0。
- **只比 prefill，不比 decode 采样**。SP 仅在 prefill 生效；且采样跨进程会分叉——基线自己跑两次也会，无法区分是 SP 的错还是采样随机性。判据取 `Sampler` 入参的**原始 logits**，而不是采样出的 token id（近 tie 时会翻转）。
- **case 按 `token_range` 的分支挑**，而不是随机挑长度：T=7（切 4/3，余数在 rank0）、T=13（切 7/6）、T=512 与 4096（整除），再叠加"多请求打包"与"单条长序列"两种形态。
- **计时只计单次 prefill step**，不测端到端吞吐，避免 decode 阶段把 SP 的影响稀释成噪声。
- **第 ① 层用 gloo 在 CPU 上跑**，不占卡，便于把 2/3/4 卡的各种 `N` 与余数组合快速扫完。测试里带 cp 组只是为了复用同一套 `GroupCoordinator` 原语、顺带验证跨组行为，SP 本身只用 TP 组。

### 5.2 指标口径

| 指标 | 定义 | 用途 | 判据 |
| --- | --- | --- | --- |
| `max\|Δ\|` | 基线与 SP 的 last-token logits 逐元素作差的最大绝对值 | 正确性**主指标** | 必须为 0（逐位一致） |
| `top1` | 两边 argmax 是否同一个 token | 交叉检查 | 一致（`max\|Δ\|=0` 时必然成立） |
| 激活峰值 | `reset_peak_memory_stats()` 后跑一次 prefill，取 `allocated_bytes.all.peak − 调用前 current` | 收益**主指标** | 越小越好，但要换算成占整机显存的比例才看得出量级 |
| prefill 时延 | 单次 prefill step，3 次取中位数 | 成本 | 越小越好 |
| KV blocks | `allocate_kv_cache` 算出的块数 | **只作观测项** | 不作为 SP 效果判据（原因见 5.6.b） |
| 通信量 | 按 V 折算的理论值（`all_reduce ≈ 2V`、`all_gather ≈ V`） | 解释时延变化 | 不单独作判据 |

口径上的三个坑：

- 峰值必须**减掉调用前 current**，否则会把权重和 KV 池一起算进"激活"。
- 必须显式 `reset_peak_memory_stats()`，否则历史峰值会污染读数。
- 时延取中位数而不是均值/最小值，避开首次编译与偶发抖动。

### 5.3 正确性（第 ③ 层）

`case` 列写成 `3+3+1` 表示这一次 prefill 里打包了三条请求、prompt 长度分别是 3/3/1（`cu_seqlens_q = [0,3,6,7]`，T=7）。**加号是打包拼接，不是乘法**：数字个数 = 请求数，它们之和 = T。多请求打包才会让 `last_indices` 取到多行，单条 case 则退化成只有一个分界点。

| case | 总 token | max\|Δ\| 基线自身两次 | max\|Δ\| 基线 vs SP | top1 |
| --- | --- | --- | --- | --- |
| 3+3+1 | 7 | 0 | 0 | ✅ |
| 5+5+3 | 13 | 0 | 0 | ✅ |
| 256+256 | 512 | 0 | 0 | ✅ |
| 单条 | 4096 | 0 | 0 | ✅ |

覆盖整除与不整除，均逐位一致。第 ① 层：gloo 下 `(tp, cp)` 组合 `(2,1)/(2,2)/(4,2)`、T=7/8/13 全过。

### 5.4 embedding 分片语义（第 ② 层）

合成权重，词表 8 / hidden 4、第 v 行填 v，8 个 token（id 0~7）、TP=2，所以期望 rank0 拿 token 0~3、rank1 拿 token 4~7：

| 版本 | rank0 | rank1 | 期望 |
| --- | --- | --- | --- |
| 修复前 | `[4,6,8,10]` | `[4,6,8,10]` | `[0,1,2,3]` / `[4,5,6,7]` |
| 修复后 | `[0,1,2,3]` | `[4,5,6,7]` | 一致 |

修复前的值满足 `j + (T/2 + j)`：第 j 行由"token j 在 rank0 词表段的部分"与"token T/2+j 在 rank1 词表段的部分"相加而成（第 j 行全填 `4+2j`），即 2.1 描述的错法。两个 rank 拿到同一份错误结果，所以两条序列会解出相同输出——这正是该 bug 最隐蔽的地方。

这类错误形状完全正常、logits 照常产出，只有比对数值才抓得到——这也是为什么要单独设第 ② 层。

### 5.5 性能与显存（第 ④ 层）

| 总 token（请求构成） | 基线 prefill | SP prefill | 时延 | 激活峰值 基线 → SP | KV blocks |
| --- | --- | --- | --- | --- | --- |
| 2048（1×2048） | 56.97 ms | 71.90 ms | +26.2% | 145.0 → 137.0 MB（−8.0 MB，−5.5%） | 2148 / 2148 |
| 8192（2×4096） | 89.57 ms | 94.83 ms | +5.9% | 252.2 → 220.2 MB（−32.0 MB，−12.7%） | 2148 / 2147 |
| 16384（4×4096） | 173.47 ms | 183.52 ms | +5.8% | 472.3 → 408.3 MB（−64.0 MB，−13.5%） | 2148 / 2148 |

### 5.6 结论

a. **正确性达标**，但显存收益很薄。绝对量约每 1k token 4 MB（对应两个 T×H 张量），落到整机 30.8 GB 上只有 0.03~0.2%。

b. **换不成容量**。`allocate_kv_cache` 在权重加载完、尚未跑任何 forward 时就按当时的 peak（≈权重占用）算死 `num_kvcache_blocks`，SP 省的激活那时还不存在，实测块数未变（16K 档省下的 64 MB 若全部能换成 KV，约合 21 个 block、+1%）。所以不要用"KV 块是否变多"判断 SP 是否有效；唯一变现路径是"同 T 峰值更低 → 可以开更大的 `max_num_batched_tokens` 而不 OOM"。

c. **时延净亏**。8K/16K 档 +5.8%，2K 档 +26%（固定通信延迟摊不薄）。

d. **+5.8% 里约 1/3 是实现成本**。按 V 折算：基线每层两次 `all_reduce` = 4V；SP 每层是两对 `(all_gather V + 伪 reduce_scatter 2V)` = 6V。6V 里多出来的 2V 来自"伪 reduce_scatter"，正好占 1/3。换原生 `reduce_scatter_tensor` 可回到 4V，但需要把 `token_range` 与 `all_gather` 的分片规则一起改成等长块语义（原生只支持等长块，余数会堆到尾部 rank，`T < N` 时还会出现 0 行 chunk），属方案级改动。

e. **当前配置下净收益≈0**。这符合 SP 的定位（长序列 prefill、激活占比高）：要转正需要更大模型、32K+ 长 T，或压低 KV 分配让激活占比变高。

## 六、遗留与后续

遗留：

- `T < N` 时 `token_range` 会产生 0 行区间（如 T=3、N=8 → `sizes=[1,1,1,0,...]`），`all_gather` 会对空 tensor 发起集合通信，HCCL 下行为未验证。建议加兜底或补用例。
- 仓库自带的 [ut/test_parallel_state.py](file:///home/d00949339/workspace/nano-vllm-ascend/ut/test_parallel_state.py) 在这个容器里会超时失败：用例把 rendezvous 超时写死 10s，而单次 `import torch_npu` 就要 7~8s，4 进程并发启动必然超时。属测试基建问题，非代码缺陷；本次 ① 层验证改用容器内 `/tmp/nano_shim/gloo_check.py` 完成。
- [attention.py](file:///home/d00949339/workspace/nano-vllm-ascend/nanovllm/layers/attention.py) prefill 分支的 q/k/v 在 CANN 9.1 下需要 `.contiguous()`（`FusedInferAttentionScore` 报 `EZ0037`），基线与 SP 都受影响，尚未提交。
- 未覆盖：32K 长序列、TP=4/8、profiler 通信占比拆解。

v2 方向（按收益排序）：

1. 原生 `reduce_scatter_tensor` + 等长块语义：Row 出口与 embedding 入口都从 2V 降到 ≈V，是当前时延亏损的主要来源。
2. 异步化：`async_op=True` + `work.wait()`，把 gather 塞进 RMSNorm 的计算间隙。
3. MoE / 多模态支持：前者要把专家循环改成 block 级 SP，后者要接管 `inputs_embeds` 入口。
