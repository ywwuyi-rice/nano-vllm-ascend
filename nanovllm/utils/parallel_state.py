import torch
import torch.distributed as dist
import torch.nn.functional as F


class GroupCoordinator:
    def __init__(self,group_ranks:list[int],group:dist.ProcessGroup|None):
        # 通信组的rank列表，通信组
        self.group_ranks = group_ranks
        self.device_group = group
        self.size = len(group_ranks)
        self.rank = group_ranks.index(dist.get_rank())

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.size ==1:
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def gather(self,input_:torch.Tensor,gather_list: list[torch.Tensor]|None=None,dst:int=0):
        if self.size == 1:
            if gather_list is not None:
                gather_list[0] = input_
            return input_
        dist.gather(input_,gather_list,self.group_ranks[dst],self.device_group)
        return input_

    def _token_range(self,total_len:int) ->tuple[int,int]:
        base ,rem = divmod(total_len, self.size)
        start = self.rank*base+min(self.rank,rem) #把余数均分给前面的rank
        end = start+base+(1 if self.rank < rem else 0)
        return start,end

    def reduce_scatter(self, input_:torch.Tensor) ->torch.Tensor:
        if self.size == 1:
            return input_
        self.all_reduce(input_)
        start,end = self._token_range(input_.shape[0])
        # clone：切片是 view，会拖住完整 T×H 存储，抵消 SP 的显存收益
        output_ = input_[start:end].clone()
        return output_
        
        
    def all_gather(self,input_:torch.Tensor,total_len:int) ->torch.Tensor:
        if self.size == 1:
            return input_
        start,end = self._token_range(total_len)
        base, rem = divmod(total_len,self.size)
        sizes = [base+1 if r< rem else base for r in range(self.size)]
        max_len = base+(1 if rem else 0)
        my_len = end-start
        assert my_len == sizes[self.rank]
        output_ = F.pad(input_,(0,0,0,max_len-my_len))
        bufs=[torch.empty_like(output_) for _ in range(self.size)]
        dist.all_gather(bufs,output_,self.device_group)
        return torch.cat([b[:s] for b,s in zip(bufs,sizes)],dim=0)





_TP: GroupCoordinator | None = None
_CP: GroupCoordinator | None = None


def init_distributed_environment(world_size, rank ,hccl_port):
    dist.init_process_group(backend= "hccl",world_size=world_size,rank=rank,
                            init_method=f"tcp://localhost:{hccl_port}")
    torch.npu.set_device(rank)


def init_model_parallel(tensor_parallel_size: int,context_parallel_size: int):
    world_size = tensor_parallel_size * context_parallel_size
    assert world_size == dist.get_world_size(),"world size not match"
    rank = dist.get_rank()
    all_ranks = torch.arange(world_size).reshape(-1,tensor_parallel_size)

    global _TP
    assert _TP is None ,"tp group already initialized"
    for cp_row in range(context_parallel_size):
        ranks = all_ranks[cp_row].tolist()
        group = dist.new_group(ranks)
        if rank in ranks:
            _TP = GroupCoordinator(ranks,group)
    
    global _CP
    assert _CP is None ,"cp group already initialized"
    if context_parallel_size > 1:
        for tp_col in range(tensor_parallel_size):
            ranks = all_ranks[:,tp_col].tolist()
            group = dist.new_group(ranks)
            if rank in ranks:
                _CP = GroupCoordinator(ranks,group)
    else:
        _CP = GroupCoordinator([rank],None)

def get_tp_group() ->GroupCoordinator:
    assert _TP is not None ,"tp group not initialized"
    return _TP

def get_cp_group() ->GroupCoordinator:
    assert _CP is not None ,"cp group not initialized"
    return _CP

    