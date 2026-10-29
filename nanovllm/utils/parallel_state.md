# parallel_state.py：多并行策略的通信组管理

## 背景：为什么需要它

原先的 nano-vllm 中只有 tp，没有其他的并行化策略。如果想自己实现其他的并行化策略，就需要在这个类中添加相关的函数。

目前计划实现 sp、cp。为了让多个并行化叠加，因此需要一个通信组来管理多个并行化策略。这个就是 parallel_state.py 的作用。

相比原版，目前要实现 sp，缺失的部分从 GroupCoordinator 开始。

## 核心抽象：GroupCoordinator

GroupCoordinator——通信组类：一个并行化策略对应一个通信组，目前有 tp_group、cp_group。（SP 不单独建组：SP 的 token 切分直接复用 TP 组，调用 get_tp_group() 即可。）

里面有 rank、size（进程数）、group_ranks（组内的全局 rank）、device_group（通信组）等属性：

```python
class GroupCoordinator:
    def __init__(self, group_ranks: list[int], group: dist.ProcessGroup | None):
        self.group_ranks = group_ranks                    # 组内全局 rank 列表（有序）
        self.device_group = group                         # torch 进程组；cp=1 占位时可为 None
        self.size = len(group_ranks)                      # 组内进程数
        self.rank = group_ranks.index(dist.get_rank())    # 本进程在组内的 rank
```

这里刻意只接两个参数：size 和 rank 都由 group_ranks 推导，不需要外部传入——单一事实来源，避免参数传串。

GroupCoordinator 的剩余函数都是各种通信原语，例如 all_reduce、all_gather、reduce_scatter 等。这些函数供给其他并行化策略调用，所以先看 parallel_state.py 里的其他内容。

## 一、模块级：环境初始化与建组

### 1. 全局单例 _TP / _CP

保存“本进程所属”的 TP / CP 通信组，初始为 None，只能通过 getter 访问。之所以用全局单例：模型各层（linear.py、embed_head.py、各模型文件）在任意位置都需要知道自己属于哪个组，逐个传参不现实。

```python
_TP: GroupCoordinator | None = None
_CP: GroupCoordinator | None = None
```

### 2. init_distributed_environment

只做两件事：初始化 hccl 进程组 + 绑定当前设备。

```python
def init_distributed_environment(world_size, rank, hccl_port):
    dist.init_process_group(backend="hccl", world_size=world_size, rank=rank,
                            init_method=f"tcp://localhost:{hccl_port}")
    torch.npu.set_device(rank)
```

为什么单独抽出来：把“后端相关”（hccl / npu）与“建组逻辑”（后端无关）解耦。好处是单测可以自己用 gloo 初始化，然后直接调 init_model_parallel，纯 CPU 就能验证建组正确性。

### 3. init_model_parallel

#### 3.1 rank 布局

第一步先校验进程数，然后构造一张 rank 表：

```python
def init_model_parallel(tensor_parallel_size: int, context_parallel_size: int):
    world_size = tensor_parallel_size * context_parallel_size
    assert world_size == dist.get_world_size(), "world size not match"
    rank = dist.get_rank()
    all_ranks = torch.arange(world_size).reshape(-1, tensor_parallel_size)
```

核心是 rank 布局约定：

    world_rank = cp_rank * tp_size + tp_rank

即 TP 在内圈（组内 rank 相邻）、CP 在外圈。TP 通信最频繁，相邻卡之间的 HCCS 带宽最高，所以占最优拓扑。reshape 之后，行 = TP 组，列 = CP 组：

          列0       列1
    行0  [0,  1]     ← cp_rank=0 的 TP 组
    行1  [2,  3]     ← cp_rank=1 的 TP 组

    all_ranks[cp_rank]        → TP 组
    all_ranks[:, tp_rank]     → CP 组
    cp_rank, tp_rank = divmod(rank, tp_size)

#### 3.2 建 TP 组

遍历所有行，每个进程都参与调 new_group，但只有组成员保存它：

```python
    global _TP
    assert _TP is None, "tp group already initialized"
    for cp_row in range(context_parallel_size):
        ranks = all_ranks[cp_row].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP = GroupCoordinator(ranks, group)
```

#### 3.3 建 CP 组

同理遍历所有列；cp_size == 1 时不建组，直接用单成员占位：

```python
    global _CP
    assert _CP is None, "cp group already initialized"
    if context_parallel_size > 1:
        for tp_col in range(tensor_parallel_size):
            ranks = all_ranks[:, tp_col].tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                _CP = GroupCoordinator(ranks, group)
    else:
        _CP = GroupCoordinator([rank], None)
```

#### 3.4 两个关键点

a. new_group 是集合调用。所有进程必须以相同顺序、相同 ranks 参数调用每一个 new_group，即使自己不属于该组也要调（内部要做 rendezvous 握手）。只有“是否保存返回值”是本地的：if rank in ranks。如果每个 rank 只建自己的组，HCCL 下大概率挂死。

b. cp_size == 1 时为什么必须占位：torch 中 group=None 的语义是 WORLD 组。如果不做单成员占位，将来 CP 代码在 cp=1 时调 all_reduce 不会报错，而是静默做了一次全网求和——结果错且不报错。

### 4. get_tp_group / get_cp_group

断言组非 None 后返回：

```python
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tp group not initialized"
    return _TP

def get_cp_group() -> GroupCoordinator:
    assert _CP is not None, "cp group not initialized"
    return _CP
```

注意判断的是 _TP is not None，而不是 dist.is_initialized()——dist 初始化了并不代表组已经建好，后者挡不住“组还没建就调用”的场景。

## 二、GroupCoordinator 的通信原语

### 0. 统一约定：size == 1 短路

每个原语开头都做 size == 1 短路。原因同 3.4：单成员组的 device_group 可能是 None，而 torch 里 group=None 等于 WORLD 组，不短路就会静默发起全网通信，轻则结果错误，重则挂死。

### 1. all_reduce

语义：组内各 rank 的 tensor 逐元素求和，结果原地写回每个人。

```python
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_
```

torch 该接口无返回值、原地修改，封装返回 x 只是为了支持链式写法 y = group.all_reduce(y)。用途：纯 TP 下 RowParallelLinear 的部分和合并；SP 的 reduce_scatter 内部也复用它。

### 2. gather

语义：只有 dst 这一个 rank 收齐全组 tensor 列表，其他 rank 只参与发送。

```python
    def gather(self, input_: torch.Tensor, gather_list: list[torch.Tensor] | None = None, dst: int = 0):
        if self.size == 1:
            if gather_list is not None:
                gather_list[0] = input_
            return input_
        dist.gather(input_, gather_list, self.group_ranks[dst], self.device_group)
        return input_
```

细节：dst 是“组内 rank”，而 torch 的 dist.gather 要求“全局 rank”，所以需要 self.group_ranks[dst] 做一次翻译。size==1 时把 x 写进 gather_list[0]，履行“调用后 dst 的列表里有值”的契约，否则调用方会读到 empty 的垃圾值。用途：ParallelLMHead 收集各 rank 的 logits（只有 rank0 需要）。

### 3. token_range

SP 的 token 切分公式，纯本地计算、零通信：

```python
    def token_range(self, total_len: int) -> tuple[int, int]:
        base, rem = divmod(total_len, self.size)
        start = self.rank * base + min(self.rank, rem)   # 把余数均分给前面的 rank
        end = start + base + (1 if self.rank < rem else 0)
        return start, end
```

含义：把 T 个 token 连续切给 N 个 rank，多出的 rem 个每人最多分 1 个、给最前面的 rem 个。

性质：所有区间首尾相接、无重叠、长度之和恰为 T——所以按 rank 顺序 cat 就能无损还原。所有 rank 用同一公式、对同一个 T 各自计算，天然对齐，不需要任何通信交换元数据。前提是 T 必须全网一致（来自调度阶段广播的 cu_seqlens）。

### 4. reduce_scatter

语义：组内求和后，每个 rank 只保留属于自己的 1/N 连续切片，数学上等价于 all_reduce + 本地 slice。

```python
    def reduce_scatter(self, input_: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return input_
        self.all_reduce(input_)
        start, end = self.token_range(input_.shape[0])
        # clone：切片是 view，会拖住完整 T×H 存储，抵消 SP 的显存收益
        output_ = input_[start:end].clone()
        return output_
```

两个注意点：

a. T 直接取 x.shape[0]：这里的输入是完整序列，与 all_gather 的输入是碎片相反。

b. 必须 .clone()：切片是 view，底层仍引用完整的 T×H 存储，父 tensor 释放不掉，SP 的激活显存收益会被抵消。

为什么 v1 用模拟实现：原生 reduce_scatter 要求 T 能被 N 整除，而打包后的 token 数不保证。通信量代价是 all_reduce 约 2V（V = T×hidden），高于原生的约 V，v2 再优化。

### 5. all_gather

语义：各 rank 手里不等长的 shard，拼回完整 T 个 token。

```python
    def all_gather(self, input_: torch.Tensor, total_len: int) -> torch.Tensor:
        if self.size == 1:
            return input_
        start, end = self.token_range(total_len)
        base, rem = divmod(total_len, self.size)
        sizes = [base + 1 if r < rem else base for r in range(self.size)]
        max_len = base + (1 if rem else 0)
        my_len = end - start
        assert my_len == sizes[self.rank]
        output_ = F.pad(input_, (0, 0, 0, max_len - my_len))
        bufs = [torch.empty_like(output_) for _ in range(self.size)]
        dist.all_gather(bufs, output_, self.device_group)
        return torch.cat([b[:s] for b, s in zip(bufs, sizes)], dim=0)
```

四步：本地算各 rank 的真实长度 sizes；尾部 pad 到 max_len（唯一目的是满足原生 all_gather “所有输入同 shape”的要求）；dist.all_gather 取回 N 份；每份按 sizes[r] 截断，再按 rank 顺序 cat。

三个注意点：

a. total_len 必须由调用方传入，绝不能从 input_.shape[0] 推：进来的是 shard（只有 T/N 行），推出来的是碎片长度，且各 rank 长度不同会导致 pad 目标不一致——要么 shape 不匹配挂死，要么结果错位。

b. 只能尾部 pad，因为截断用的是 b[:s]，依赖多余数据在尾部。

c. assert my_len == sizes[self.rank] 自检：验证 token_range 与 sizes 两套推导一致。

通信量约等于 V（只多传不到 N 行），已接近理论下限。和 gather 的区别：gather 只有 dst 收，all_gather 是人人一份。

### 6. 通信量小结

（V = T × hidden_size，N = 组大小）

| 方案 | 每 rank 通信量 | 说明 |
| --- | --- | --- |
| all_reduce | ≈ 2V | 环形实现，reduce + allgather 两阶段 |
| reduce_scatter（v1 模拟） | ≈ 2V | all_reduce + 本地切片 |
| reduce_scatter（原生，v2） | ≈ V | 要求 T 整除 N |
| all_gather（v1，pad 到 max_len） | ≈ V | 仅多传 < N 行 |
| 反例：pad 到 T 再求和 | ≈ N·V | 零也占带宽，为位置信息付 N 倍税 |

## 三、SP 的数据流

以 prefill 的一个 DecoderLayer 为例：

    token-sharded（T/N 行）
        │ all_gather(total_len=T)          ← 进 Column 层前恢复完整序列
        ▼
    完整 T 行 → qkv_proj → RoPE → attention → o_proj
        │ reduce_scatter                  ← 出 Row 层后切回自己那 1/N
        ▼
    token-sharded → RMSNorm → MLP 同样模式

要点：attention / RoPE / RMSNorm 全在“完整 T”区间工作，本身零改动；SP 的改动只发生在进出边界的通信上。decode 阶段 batch 小、收益低，不启用 SP。

## 四、未来展望

### 1. 新增一种并行策略的统一步骤（以 DP 为例）

parallel_state 的设计是“一个策略 = 一个 GroupCoordinator 单例”，接入新策略只需要四步。

a. 扩展 rank 布局。新维度放最外层（通信最少），CP、TP 的相对顺序不变：

    world_rank = dp_rank * (cp_size * tp_size) + cp_rank * tp_size + tp_rank

对应 reshape 从二维变三维（之前讨论过的“要不要 reshape 成 (-1, cp, tp)”，就是为这一天准备的）：

```python
all_ranks = torch.arange(world_size).reshape(-1, cp_size, tp_size)
# 第一维 = DP 副本，第二维 = CP，第三维 = TP
```

b. 按同样的集合调用模式建组，遍历所有 DP 副本，并加一组单例和 getter：

```python
_DP: GroupCoordinator | None = None

for dp_idx in range(dp_size):
    ranks = all_ranks[dp_idx].flatten().tolist()
    group = dist.new_group(ranks)          # 所有进程都要调，不只是组成员
    if rank in ranks:                      # 只有成员保存
        _DP = GroupCoordinator(ranks, group)

def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "dp group not initialized"
    return _DP
```

c. 调用点使用对应的 getter：每个并行策略只认自己那个组，避免“该走 TP 的通信走了 WORLD”。

d. 并行策略之间是笛卡尔积：world_size = 各维乘积。init_model_parallel 的参数列表会随策略数量增长，后续建议改成传一个 ParallelConfig 对象。

### 2. 需要新增的通信原语

| 策略 | 缺少的原语 | 说明 |
| --- | --- | --- |
| EP / MoE | all_to_all | token 按专家分发、结果回收 |
| PP | send / recv | 相邻 stage 之间传 activation |
| CP | 带 dim 参数的 all_gather | CP 收集的是 KV（序列/head 维），与 SP 的 token 维不是一回事 |
| 通用 | broadcast | rank0 的调度/采样信息广播给其他 rank |

新增原语的三条硬约束（与现有实现保持一致）：

a. 开头一律做 size == 1 短路；

b. 显式传 group=，绝不让 group 落到 None（None 等于 WORLD）；

c. 集合调用的参数（shape、split sizes）全网一致，且尽量由本地元数据推导，避免为“协商参数”额外通信。

### 3. 原语的性能演进（v1 → v2）

目前 reduce_scatter / all_gather 是模拟实现：前者 all_reduce + slice，后者 pad + all_gather + cat。演进方向：

a. 等分化：入口把 T padding 到 N 的倍数，即可使用原生 reduce_scatter_tensor / all_gather_into_tensor，通信量从 ≈2V 降到 ≈V；

b. 异步与重叠：async_op=True + work.wait()，把通信塞进计算间隙（CP 的 KV all_gather 尤其需要）；

c. 图模式适配：SP 首版限定 enforce_eager，等 torchair fullgraph 对 dist 算子支持稳定后再打开；

d. 拓扑：保持“通信最频繁的维放最内层”（现在是 TP），新增 DP 时 DP 放最外层。

### 4. 什么时候该动 parallel_state.py

判断标准很简单：

- 出现“新的 rank 子集需要独立做集合通信” → 加新组（单例 + getter + 建组循环）；
- 出现“现有原语语义不够用”（维度不同、切分不均、需要单向通信） → 加新原语；
- 只是改变某个维度的大小或开关 → 只改 config 和调用点，不动这里。

这也意味着 parallel_state.py 会长期是一个“薄而稳定”的层：策略实现的复杂度应该在模型层，而不是堆在通信层里。
