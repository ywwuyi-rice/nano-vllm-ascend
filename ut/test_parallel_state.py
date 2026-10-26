from datetime import timedelta
from nanovllm.utils.parallel_state import init_model_parallel, get_tp_group, get_cp_group

import torch 
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank,port,tp_size,cp_size,case):
    dist.init_process_group(backend= "gloo",world_size=tp_size*cp_size,rank=rank,
                            init_method=f"tcp://localhost:{port}",timeout = timedelta(seconds=10))
    init_model_parallel(tp_size,cp_size)
    tp = get_tp_group()
    cp = get_cp_group()
    

    if case =="metadata":
        _check_metadata(rank,tp_size,cp_size,tp,cp)
    elif case == "all_reduce":
        _check_all_reduce(rank,tp,cp)
    dist.destroy_process_group()


def _check_metadata(rank,tp_size,cp_size,tp,cp):
    assert tp.size == tp_size
    assert cp.size == cp_size
    assert tp.group_ranks == list(range(rank//tp_size*tp_size,rank//tp_size*tp_size+tp_size))
    assert cp.group_ranks == list(range(rank % tp_size, rank % tp_size + (cp_size - 1) * tp_size + 1, tp_size))
    assert tp.rank == rank % tp_size
    assert cp.rank == rank // tp_size

def _check_all_reduce(rank,tp,cp):
    input_tensor = torch.tensor([float(rank+1)])
    tp.all_reduce(input_tensor)
    expect_value = float(sum(r + 1 for r in tp.group_ranks))
    assert input_tensor.item() == expect_value, f"rank={rank} TP:got {input_tensor.item()}"

    x = torch.tensor([float(rank + 1)])
    cp.all_reduce(x)
    expected = float(sum(r + 1 for r in cp.group_ranks))
    assert x.item() == expected, f"rank={rank} CP: got {x.item()}, expect {expected}"
    
if __name__ == "__main__":
    for tp, cp, case, port in [(2,2,"metadata",29500),(2,2,"all_reduce",29501),
                                (2,1,"metadata",29502),(2,1,"all_reduce",29503),
                                (2,4,"metadata",29504),(4,2,"metadata",29505)]:
        mp.spawn(_worker, args=(port,tp,cp,case), nprocs=tp*cp, join=True)
    print("all passed")