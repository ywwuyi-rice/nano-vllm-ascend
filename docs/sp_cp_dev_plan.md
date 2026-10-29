# SP / CP 并行特性开发计划

> 状态跟踪文档（用户实现 / AI review 模式）。最后更新：2026-10-29，**A1/A2/B1~B6 代码已完成，待 B2 单测 + A3/B7 有卡环境验收与调参**。
>
> 代码状态：SP 全部接线已在本机完成（仅 `py_compile` 通过，无运行时验证）；上一次 push 的基线为 commit `a004f47`。

## 1. 背景

项目当前仅有 TP：`dist.get_rank()/get_world_size()` 散落在 `nanovllm/layers/linear.py`、`layers/embed_head.py`、各模型文件中，无进程组抽象。开发路线：先 SP（Megatron 式 TP×SP，仅 prefill 生效），后 CP（长上下文）。

关键架构约束：
- rank0 为主进程（调度、采样），其余 rank 由 `llm_engine.py` spawn、经共享内存接收调用
- 模型文件 head 数在构造期确定，因此 `init_model_parallel` 必须在构造 model 之前调用

## 2. 总体设计

### M1：parallel_state 基础设施（✅ 已完成）
- `GroupCoordinator`：进程组封装（`rank/size/group_ranks` + `all_reduce/gather`，`size==1` 短路）
- rank 布局：`world_rank = cp_rank * tp_size + tp_rank`（TP 组内 rank 相邻，通信域更优）
- `all_ranks = arange(world).reshape(-1, tp_size)`：**行 = TP 组**（行号 = cp_rank），**列 = CP 组**（列号 = tp_rank）
- `init_distributed_environment`（hccl 后端相关）与 `init_model_parallel`（后端无关）分离，后者可在 gloo 下测试
- `gather` 封装注意：torch 的 `dist.gather` 的 dst 参数要求**全局 rank**，封装内用 `group_ranks[dst]` 做了组内→全局翻译

### M2：SP（TP×SP，仅 prefill）
- 原理：`RowParallelLinear` 的 all_reduce → reduce_scatter（token 维切分激活）；Column 并行层前置 all_gather；RMSNorm/attention/rope/MoE router 零改动
- embedding 入口用 `token_range` 切 `input_ids`（保留词表维的 `all_reduce`）；lm_head 入口前置 `all_gather` 以支撑全局 `last_indices`
- **SP 不需要新建进程组**，复用 TP 组
- v1 用 all_reduce+slice / gather+cat 模拟（免 padding），后续再优化
- 通过 `context.is_prefill` 分支，decode 不启用；首版限 `enforce_eager`（torchair fullgraph 捕获 dist 算子单独适配）

### M3：CP（长上下文）
- prefill：q chunk 本地计算、KV 跨 CP 组 all_gather（data-parallel attention 路线）；需重构 atten_mask（现硬编码 2048）
- decode：推荐 KV 全 rank 复制方案（prefill 段按 cp 切 + decode 段共享，零跨卡通信）
- 采样：每序列 last token 落在最后一个 chunk，all_gather 到 rank0

### M4：组合验证 + bench
- ut 正确性测试（greedy 逐 token 一致为准绳）+ bench 各组合吞吐/显存对比

## 3. 当前进度

### ✅ A1：parallel_state + gloo 测试（已完成）
- [parallel_state.py](../nanovllm/utils/parallel_state.py) 全部完成：GroupCoordinator（`self.rank`、all_reduce/gather 封装、size==1 短路）、按行/列遍历建组（new_group 集合契约正确）、cp=1 占位组、getter assert
- [ut/test_parallel_state.py](../ut/test_parallel_state.py) **六用例全绿**（本机 WSL gloo 验证）：2×2 metadata/all_reduce、2×1 退化、2×4、4×2 非方阵布局
- ⚠️ 测试文件当前用直接 `from nanovllm...` import，**本机（无 torch_npu）跑不了**，需在有卡环境执行；如需本机跑，可临时改回 importlib 按路径加载

### ✅ A2：接线（已完成，2026-10-22）
全部 `dist.get_world_size()/get_rank()` 已替换为 `get_tp_group().size/.rank`：
- [model_runner.py](../nanovllm/engine/model_runner.py) L42-43：`init_distributed_environment(world_size, rank, hccl_port)` + `init_model_parallel(tp, 1)`（含 CP TODO 注释），在构造 model 之前
- [linear.py](../nanovllm/layers/linear.py)：6 处（LinearBase 的 tp_rank/tp_size、三个子类 tp_size、Row 的 all_reduce 走 TP 组）
- [embed_head.py](../nanovllm/layers/embed_head.py)：4 处（含 embedding all_reduce、LMHead gather 走 TP 组）
- 6 个模型文件（llama/qwen3/qwen3_moe/mini_cpm4/qwen3_vl/qwen2_5_omni）：各 2 处
- 验证：残留 `dist.*` 扫描仅剩 parallel_state.py 自身 5 处（合法）；10 文件语法检查通过

### ✅ B2：SP 通信原语（已完成，未测）
[parallel_state.py](../nanovllm/utils/parallel_state.py) 新增：

| 方法 | 实现要点 |
|---|---|
| `_token_range(total)` | `base, rem = divmod(total, size)`；`start = rank*base + min(rank, rem)`；余数分给靠前 rank，连续切分 |
| `reduce_scatter(x)` | `all_reduce(x)` + 本地切片 + **`.clone()`**（避免视图拖住全量 activation） |
| `all_gather(shard, total)` | `max_len = base + (1 if rem else 0)` → `F.pad` 尾部补齐 → 原生 `all_gather` → 按 `sizes` 截断 → `cat(dim=0)` |

- `total` 必须由调用方传入（进函数时 input 已是 shard，不能反推）
- `sizes` / 区间全部本地推导，零通信交换元数据
- 已含 `assert my_len == sizes[self.rank]` 自检
- ⏳ **缺单测**：gloo 用例需覆盖 T 不整除（T=7, N=2/3）与整除（T=8）

### ✅ B1：开关链路（已完成，未测）
`--enable-sp` 从 CLI 一路到 `set_context`，四个入口都已打通：

- [config.py](../nanovllm/config.py) L22 加 `enable_sp: bool = False`；L41-47 断言：`tensor_parallel_size > 1`、`enforce_eager == True`、架构名不含 `Moe`、`model_type` 不属于 `{qwen3_vl, qwen2_5_omni}`
- [arg_utils.py](../nanovllm/utils/arg_utils.py) 加 `enable_sp` 字段与 `--enable-sp/-sp`（argparse 自动映射到 `dest=enable_sp`，`from_cli_args` 无需改动）
- [api_server.py](../nanovllm/v1/entrypoints/openai/api_server.py) L108 显式传入 `AsyncLLMEngine(enable_sp=args.enable_sp)`（原代码读 `args.enable_sp` 却没传，属静默失效）
- [bench.py](../bench/bench.py) 的 `LLM(...)` 同样补传
- 注意：多模态断言不能用 `self.is_multimodal`——它在 `__post_init__` 时仍是 `False`，由 model_runner 事后回填，只能查 `hf_config`

### ✅ B3/B4：线性层接线（已完成，未测）
- [linear.py](../nanovllm/layers/linear.py) L73-77 `ColumnParallelLinear.forward`：`enable_sp and is_prefill` 时 `all_gather(x, ctx.num_tokens)` 后做 `F.linear`；`MergedColumnParallelLinear` / `QKVParallelLinear` 继承此类，改一处全覆盖
- 同文件 L153-161 `RowParallelLinear.forward`：SP+prefill 走 `reduce_scatter(y)`，否则维持 `all_reduce(y)`
- bias 仍只在 `tp_rank == 0` 加，`reduce_scatter` 内部先 all_reduce，语义不变

### ✅ B5：embedding / lm_head 端点（已完成，未测）
- [embed_head.py](../nanovllm/layers/embed_head.py) L34-47 `VocabParallelEmbedding.forward`：**不能先切 `input_ids`**。词表维 `all_reduce` 按行号把各 rank 的同一行相加，各 rank 必须持同一批 token；先切会让 rank0 的第 i 行（token i）和 rank1 的第 i 行（token T/N+i）相加，得到两个无关 token 之和。因此保持全量 T 算完 `all_reduce`，出口再按 `token_range` 切片并 `.clone()`（视图会拖住全量存储）
- 成本：入口是"全量计算 + 全量归约（≈2V）"，与 Megatron `reduce_scatter_embeddings` 同思路；v2 换原生 reduce_scatter 可降到 ≈V
- 同文件 L50-58 `ParallelLMHead.forward`：`all_gather` 必须在 `cu_seqlens_q[1:] - 1` 之前，因为该索引是全局坐标系
- [parallel_state.py](../nanovllm/utils/parallel_state.py) L28 `_token_range` → `token_range`（公开），保证 embedding 切片与 `all_gather`/`reduce_scatter` 使用同一套余数分配

### ✅ B6：运行时开关（已完成，未测）
- [context.py](../nanovllm/utils/context.py) L8-9 加 `enable_sp` / `num_tokens`；`set_context` 签名同步，且 `Context(...)` 由**位置参数改为关键字构造**（字段新增导致位置错位是隐藏杀手）
- [model_runner.py](../nanovllm/engine/model_runner.py) L318-319 prefill 传 `enable_sp` 与 `num_tokens=int(cu_seqlens_q[-1].item())`（只算一次，避免每层 D2H 同步）；decode 两处不传，靠默认 `False` 双保险


### ⏳ A3：NPU 验收（需有卡环境）
1. TP=1 回归：输出与重构前一致（单卡无通信，纯回归）
2. TP=2 跑 llama（或 qwen3），greedy 输出与重构前**逐 token 一致**
3. 若不一致，排查顺序：Row/Column 的 tp_rank 语义 → embedding mask 逻辑 → LMHead gather 的全局 rank 翻译

## 4. B：SP 实现计划

| 步骤 | 内容 | 状态 |
|---|---|---|
| B1 | `config.py` 加 `enable_sp`（默认 False）、`arg_utils` 加 CLI | ✅ 完成 |
| B2 | GroupCoordinator 加 `reduce_scatter` / `all_gather` | ✅ 完成，⏳ 缺单测 |
| B3 | `RowParallelLinear.forward`：SP+prefill 时 all_reduce → reduce_scatter | ✅ 完成，未测 |
| B4 | Column 系（qkv/gate_up）forward 前置 all_gather | ✅ 完成，未测 |
| B5 | embedding 入口切片（保留词表维 all_reduce）；lm_head 前置 all_gather | ✅ 完成，未测 |
| B6 | 运行时开关借 `context.is_prefill`（decode 不启用）；首版限 `enforce_eager` | ✅ 完成，未测 |
| B7 | 验收：TP=2+SP vs TP=2 greedy 逐 token 一致 + `torch.npu.memory_stats()` 激活内存对比 | ⏳ 需有卡 |

**建议顺序**：B2 单测（本机可做，需临时改回 importlib）→ A3（有卡）→ B7（有卡）

### B7 验收与调参思路（有卡执行）

**第一步：正确性**

- 两组都 `enforce_eager=True`、greedy 采样，否则对比的是"图 vs eager"而非"SP vs 无 SP"
- 先跑 T 不被 tp_size 整除的 batch（如 batch 内 7 / 13 个 token），确认 `all_gather` 的 pad/截断路径；再跑整除 batch
- 判据：TP=2 与 TP=2+SP 的输出 token ids 完全一致；不一致时按"embedding 切口 → Column gather → Row scatter → lm_head gather"顺序二分定位

**第二步：峰值是否真的下降**

- 分阶段测量：`reset_peak_memory_stats()` 后只跑一次 prefill，记录 `max_memory_allocated()`；decode 阶段单独测（SP 不参与）
- SP 只缩小 shard 区（残差流 / RMSNorm 输出），attention 与 MLP 中间量仍在完整 T，不省
- v1 的 `all_gather` 会临时分配 pad buffer + `cat` 结果（合计约 2×T×H），峰值可能被抵消——**这一步不降，后面调参没有意义**

**第三步：省下的显存能换什么**

- ⚠️ **换不到更多 KV 块**：`allocate_kv_cache` 在权重加载完、尚未跑任何 forward 时就按当时的 `allocated_bytes.all.peak`（≈ 权重占用）一次性算死 `num_kvcache_blocks`，并覆盖外部传入值。SP 省的是激活，此时还没发生，块数不会因此变化
- 因此真正可用的变现路径只有一条：**同样的 `max_num_batched_tokens` 下 prefill 峰值更低（余量更大），从而可以把 batched tokens 调得更大而不 OOM**
- 调大 batched tokens 后 KV 块数不变，运行时更容易 OOM，需同时观察峰值；不要用"块数是否变化"来判断 SP 是否有收益

**第四步：扫参定位适用区**

- 主变量 `max_num_batched_tokens`（4K/8K/16K/32K）× `tensor_parallel_size`（2/4/8），负载必须偏 prefill（长 prompt + 短输出），否则差异淹没在 decode 噪声里
- 辅助变量 `max_num_seqs`（同样 T 下长序列少请求 vs 短序列多请求）、`kvcache_block_size`
- 关注两条曲线：prefill 峰值差、同 T 下的 TTFT 差；再找"同一显存预算下能开到的最大 batched tokens"。只有"峰值降 + 可开 T 变大 + TTFT 不明显劣化"的组合才值得默认开启
- 预期：v1 每层通信量约高于基线 50%（`all_reduce`×2 → `all_gather` + 模拟 `reduce_scatter`），TP 越大摊薄越多；TP=2 + 显存宽松时应无收益

## 5. 测试要点（ut/test_parallel_state.py，曾全绿，现需有卡环境）

- gloo + `mp.spawn`，六用例：(2,2)×2、(2,1)×2、(2,4)、(4,2)——非方阵用例防"公式只在 tp==cp 时碰巧对"
- metadata 断言（用 `group_ranks` 现算预期，不硬编码）：
  - `tp.group_ranks == list(range(rank//tp*tp, rank//tp*tp+tp))`（行，连续）
  - `cp.group_ranks == list(range(rank%tp, rank%tp+(cp-1)*tp+1, tp))`（列，等差）
  - `tp.rank == rank % tp_size`、`cp.rank == rank // tp_size`
- all_reduce 断言：tensor 填 `rank+1`，预期 `sum(r+1 for r in group_ranks)`——验证通信只在本组内
- cp=1：占位组 all_reduce 后值不变（size==1 短路）
- 两轮 spawn 换端口；assert 消息带 rank；卡死 >10s = new_group 契约破坏

## 6. 踩过的坑（review 教训）

1. **`dist.new_group` 是集合调用**：所有进程必须以相同顺序、相同 ranks 列表调每一个组（不在组内也要调），否则 gloo 行为未定义 / HCCL 挂死
2. **group=None 的集合操作默认走 WORLD 组**：单成员占位组必须在 `GroupCoordinator` 内短路，否则静默全网求和
3. 模块级注解 `_TP: GroupCoordinator | None = None` 写在 class 定义前会 NameError（注解会被求值）
4. tensor 没有 `.index()` 方法，`group_ranks` 必须转 list
5. 行/列索引混淆：行号语义是 cp_rank（`rank // tp_size`），列号是 tp_rank（`rank % tp_size`）——用 `divmod(rank, tp_size)` + 断言自检
6. **`dist.gather` 的 dst 要全局 rank**（与 group 参数无关），封装层必须做 `group_ranks[dst]` 翻译
7. `python ut/test_x.py` 直接执行时 sys.path 加的是**脚本所在目录**而非 cwd；`from nanovllm...` import 还会拖 `__init__.py` 的 torch_npu/torchair 链——纯 CPU 测试需 importlib bypass
8. IDE 自动"删除未使用导入"可能误删刚加的 import（诊断延迟），留意 quick fix；同理，"确认采纳" old-code 建议会把已修好的代码写回旧版本，commit 前先 `git diff`
9. **`all_gather` 的 `total` 不能从 `input_.shape[0]` 推**：进函数时 input 已是 shard（T/N 行），必须由调用方从 cu_seqlens 传入全网一致的 T
10. **`reduce_scatter` 要 `.clone()`**：`tensor[start:end]` 是视图，底层仍持有完整 T×H 存储，会抵消 SP 的 activation 显存收益
11. **`Context` 是位置参数构造**：给 dataclass 插字段后，`Context(is_prefill, cu_seqlens_q, ...)` 会整体后移错位（`cu_seqlens_q` 落进 `enable_sp`），类型不匹配时才会报错，兼容时静默错算。已全部改为关键字构造，后续新增字段务必保持
12. **embedding 绝不能先切 token 再归约**：这里的 `all_reduce` 是词表维、按**行号**对齐求和，各 rank 必须持同一批 token。先切会让 rank0 第 i 行（token i）与 rank1 第 i 行（token T/N+i）相加，两个 rank 拿到同一份错误结果（典型症状：所有序列输出一模一样）。正确做法是全量算 + all_reduce + 出口切片；代价是入口通信 ≈2V，v2 换原生 reduce_scatter 降到 ≈V
13. **`reduce_scatter` 的输入侧不能裁**：通用场景（如 ZeRO 梯度）中每个 rank 必须贡献它对所有分块的 partial；SP 的 Row 出口之所以能"先切后算"，是因为切片维（token）与归约维（hidden）正交，这是 TP 结构给的局部特权，不能下沉为通用原语
14. **特殊模型必须拦死**：多模态走 `inputs_embeds` 会绕过 `embed_tokens`（入口没切，下游却按 shard 处理）；MoE 的专家循环里 Column gather 后每个 rank 都算出相同 full T 结果，Row 的 all_reduce 会放大 N 倍。两者都在 `config.py` 断言拦截，v2 再谈支持
15. **索引要区分"坐标系"与"地址"**：`cu_seqlens` / `last_indices` / `positions` 是坐标系，必须与张量同坐标系；`slot_mapping` / `block_tables` 是全局地址，条目顺序跟着同区间切即可。判据：`f(x_full)[s:e] == f(x_full[s:e], meta[s:e])`
16. **别用 pad 去救"行不对齐"**：曾加过 `all_reduce_shard`（pad 到 `max_len` 再归约）想解决不整除问题，但 pad 只保证**形状**一致，救不了**行语义**——各行本就对应不同 token，补齐后照样是错的。判断标准是"两 rank 的第 i 行是否同一个 token"，不是"形状是否相同"。已删除该方法
17. **`--enforce-eager` 是 `store_true` 且无 default**：argparse 默认给 False，会覆盖 dataclass 的 True，导致 `--enable-sp` 单独使用在 `Config` 断言处失败。已在 `EngineArgs.__post_init__` 里由 `enable_sp` 兜底置 True
18. **KV 块数在权重加载后就固定**：`allocate_kv_cache` 用当时的 peak（≈权重）计算并覆盖 `num_kvcache_blocks`，SP 省的激活不会转成更多块——别用"块数变化"评价 SP

## 7. 环境备忘

- 本机（WSL，无 NPU）：测试用 `/home/d00949339/MinivLLM/.venv/bin/python`（Python 3.11.16 + torch 2.9.x，无 torch_npu/CANN）
- 有卡环境：项目装 `pip install -e .` 后正常 import；测试文件无需改动
- workspace 根目录有 `pyrightconfig.json`（指向 MinivLLM venv，悬停文档用）
