# SP / CP 并行特性开发计划

> 状态跟踪文档（用户实现 / AI review 模式）。最后更新：2026-10-22，**A2 已完成，待 A3 NPU 验收**。

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
- embedding 的 all_reduce → reduce_scatter；lm_head 前置 all_gather
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
- ⚠️ 测试文件用 importlib bypass 直接按路径加载 parallel_state（绕过 `nanovllm/__init__.py` 的 torch_npu/torchair 导入链，使纯 CPU 环境可跑）。在有 NPU 的环境可直接跑，无需修改

### ✅ A2：接线（已完成，2026-10-22）
全部 `dist.get_world_size()/get_rank()` 已替换为 `get_tp_group().size/.rank`：
- [model_runner.py](../nanovllm/engine/model_runner.py) L42-43：`init_distributed_environment(world_size, rank, hccl_port)` + `init_model_parallel(tp, 1)`（含 CP TODO 注释），在构造 model 之前
- [linear.py](../nanovllm/layers/linear.py)：6 处（LinearBase 的 tp_rank/tp_size、三个子类 tp_size、Row 的 all_reduce 走 TP 组）
- [embed_head.py](../nanovllm/layers/embed_head.py)：4 处（含 embedding all_reduce、LMHead gather 走 TP 组）
- 6 个模型文件（llama/qwen3/qwen3_moe/mini_cpm4/qwen3_vl/qwen2_5_omni）：各 2 处
- 验证：残留 `dist.*` 扫描仅剩 parallel_state.py 自身 5 处（合法）；10 文件语法检查通过

### ⏳ A3：NPU 验收（下一步，需有卡环境）
1. TP=1 回归：输出与重构前一致（单卡无通信，纯回归）
2. TP=2 跑 llama（或 qwen3），greedy 输出与重构前**逐 token 一致**
3. 若不一致，排查顺序：Row/Column 的 tp_rank 语义 → embedding mask 逻辑 → LMHead gather 的全局 rank 翻译

### 4. B：SP 实现（A3 过后）
1. `config.py` 加 `enable_sp`（默认 False）
2. GroupCoordinator 加 `all_gather/reduce_scatter`（token 维；v1 用 all_reduce+slice / gather+cat 模拟免 padding）
3. `RowParallelLinear.forward`：SP+prefill 时 all_reduce → reduce_scatter
4. Column 系 forward 前置 all_gather（token 维恢复完整序列）
5. embed_head 同模式改造
6. 运行时开关借 `context.is_prefill`（decode 不启用）；首版限 enforce_eager
7. 验收：TP=2+SP vs TP=2 greedy 逐 token 一致 + `torch.npu.memory_stats()` 激活内存对比

## 5. 测试要点（ut/test_parallel_state.py，已验证）

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
8. IDE 自动"删除未使用导入"可能误删刚加的 import（诊断延迟），留意 quick fix

## 7. 环境备忘

- 本机（WSL，无 NPU）：测试用 `/home/d00949339/MinivLLM/.venv/bin/python`（Python 3.11.16 + torch 2.9.x，无 torch_npu/CANN）
- 有卡环境：项目装 `pip install -e .` 后正常 import；测试文件无需改动
- workspace 根目录有 `pyrightconfig.json`（指向 MinivLLM venv，悬停文档用）
