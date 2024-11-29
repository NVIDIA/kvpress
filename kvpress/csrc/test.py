import torch
import random
from tiny_api_cuda import update_flatten_view, update_flatten_klenN_view


def test_single_insert(head_num, head_dim):
    head_lens = []
    for _ in range(head_num):
        head_lens.append(random.randint(1, 100))
    head_lens = torch.tensor(head_lens, dtype=torch.int32, device='cuda')
    klen_sum = torch.sum(head_lens, dtype=torch.int32)
    cu_klen = torch.cumsum(head_lens, 0, dtype=torch.int32) - head_lens
    cu_klen = torch.cat([cu_klen, torch.tensor([klen_sum], dtype=torch.int32, device="cuda")], dim=0)

    head_cache_list = []
    for hl in head_lens:
        head_cache = torch.randn((hl, head_dim), dtype=torch.bfloat16, device="cuda")
        head_cache_list.append(head_cache)
    head_cache_tensor = torch.cat(head_cache_list, dim=0)
    key_state0 = torch.randn((1, head_num, 1, head_dim), dtype=torch.bfloat16, device="cuda")

    expected_cache = head_cache_list.copy()
    for i in range(head_num):
        expected_cache[i] = torch.cat([expected_cache[i], key_state0[0, i, 0, :].view(1, head_dim)], dim=0)
    expected_cache = torch.cat(expected_cache, dim=0)
    ref_new_state_0 = update_flatten_view(head_cache_tensor.view(-1, head_dim), key_state0.view(-1, head_dim), head_lens, cu_klen)

    assert torch.equal(expected_cache, ref_new_state_0)

def test_single_insertN(head_num, head_dim, klen):
    head_lens = []
    seqlen = 3810
    # for _ in range(head_num):
    #     head_lens.append(random.randint(1, 100))
    head_lens = [seqlen] * head_num
    head_lens = torch.tensor(head_lens, dtype=torch.int32, device='cuda')
    klen_sum = torch.sum(head_lens, dtype=torch.int32)
    cu_klen = torch.cumsum(head_lens, 0, dtype=torch.int32) - head_lens
    cu_klen = torch.cat([cu_klen, torch.tensor([klen_sum], dtype=torch.int32, device="cuda")], dim=0)
    # print("cu_klen", cu_klen)

    # cu_klen = torch.cumsum(head_lens, 0, dtype=torch.int32)
    # cu_klen = torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), cu_klen, ], dim=0)
    # print("cu_klen", cu_klen)
    # input("Press Enter to continue...")
    # head_cache_list = []
    # for hl in head_lens:
    #     head_cache = torch.randn((hl, head_dim), dtype=torch.bfloat16, device="cuda")
    #     head_cache_list.append(head_cache)
    # head_cache_tensor = torch.cat(head_cache_list, dim=0)
    key_state0 = torch.randn((1, head_num, klen, head_dim), dtype=torch.bfloat16, device="cuda")
    head_cache = torch.randn((1, head_num, seqlen, head_dim), dtype=torch.bfloat16, device="cuda")

    expected_cache = torch.cat([head_cache, key_state0], dim=2)
    print("expected_cache.shape", expected_cache.shape)
    expected_cache = expected_cache.view(-1, head_dim)
    print("expected_cache.shape", expected_cache.shape)

    head_cache = head_cache.view(-1, head_dim)
    # expected_cache = head_cache_list.copy()
    # for i in range(head_num):
    #     expected_cache[i] = torch.cat([expected_cache[i], key_state0[0, i, :, :].view(-1, head_dim)], dim=0)
    # expected_cache = torch.cat(expected_cache, dim=0)


    print("head_cache_tensor.shape", head_cache.shape)
    print("key_state0.shape", key_state0.shape)
    print("head_lens", head_lens)
    print("cu_klen", cu_klen)
    ref_new_state_0 = update_flatten_klenN_view(head_cache, key_state0, head_lens, cu_klen)
    print("ref_new_state_0.shape", ref_new_state_0.shape)

    assert torch.equal(expected_cache, ref_new_state_0)
    input(f"{head_num, head_dim, klen}Test passed")

def main(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    for head_num in [8]:
        for head_dim in [128]:

            # test_single_insert(head_num, head_dim)

            for klen in [1, 2, 128, 256, 512]:

                test_single_insertN(head_num, head_dim, klen)


if __name__ == "__main__":
    for seed in range(100):
        main(seed)
