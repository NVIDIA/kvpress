import torch
import random
from tiny_api_cuda import update_flatten_klenN_view


def test_single_insertN(head_num, head_dim, klen):
    head_lens = []
    seqlen = 256
    head_lens = [seqlen] * head_num
    head_lens = torch.tensor(head_lens, dtype=torch.int32, device='cuda')
    klen_sum = torch.sum(head_lens, dtype=torch.int32)
    cu_klen = torch.cumsum(head_lens, 0, dtype=torch.int32) - head_lens
    cu_klen = torch.cat([cu_klen, torch.tensor([klen_sum], dtype=torch.int32, device="cuda")], dim=0)
    key_state0 = torch.randn((1, head_num, klen, head_dim), dtype=torch.bfloat16, device="cuda")
    head_cache = torch.randn((1, head_num, seqlen, head_dim), dtype=torch.bfloat16, device="cuda")
    expected_cache = torch.cat([head_cache, key_state0], dim=2)
    expected_cache = expected_cache.view(-1, head_dim)
    head_cache = head_cache.view(-1, head_dim)
    ref_new_state_0 = update_flatten_klenN_view(head_cache, key_state0, head_lens, cu_klen)
    assert torch.equal(expected_cache, ref_new_state_0)
    print(f"{head_num, head_dim, klen}Test passed")

def main(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    for head_num in [4, 8, 16]:
        for head_dim in [128, 256, 512]:
            for klen in [1, 4, 16, 64]:
                test_single_insertN(head_num, head_dim, klen)

# unit test for cuda kernel
if __name__ == "__main__":
    for seed in range(100):
        main(seed)
