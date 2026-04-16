"""Tests for FraqtlPress."""
import torch
import pytest


def _orthogonal_basis(H, D):
    """Create a random orthogonal eigenbasis (H, D, D)."""
    U = torch.randn(H, D, D)
    for h in range(H):
        Q, _ = torch.linalg.qr(U[h])
        U[h] = Q
    return U.half()


class _FakeModule:
    def __init__(self, idx=0):
        self.layer_idx = idx


class TestCompress:
    def test_output_shape_preserved(self):
        from fraqtl_press import FraqtlPress

        H, HD = 8, 128
        basis = {0: _orthogonal_basis(H, HD)}
        press = FraqtlPress(bits=4, eigenbasis=basis)
        press._U_cache[0] = (
            basis[0].unsqueeze(0).float(),
            basis[0].unsqueeze(0).float().transpose(-2, -1).contiguous(),
        )

        K = torch.randn(1, H, 64, HD, dtype=torch.float16)
        V = torch.randn(1, H, 64, HD, dtype=torch.float16)
        K_out, V_out = press.compress(_FakeModule(0), None, K, V, None, {})

        assert K_out.shape == K.shape
        assert V_out.shape == V.shape

    def test_keys_untouched(self):
        from fraqtl_press import FraqtlPress

        H, HD = 4, 64
        basis = {0: _orthogonal_basis(H, HD)}
        press = FraqtlPress(bits=4, eigenbasis=basis)
        press._U_cache[0] = (
            basis[0].unsqueeze(0).float(),
            basis[0].unsqueeze(0).float().transpose(-2, -1).contiguous(),
        )

        K = torch.randn(1, H, 32, HD, dtype=torch.float16)
        V = torch.randn(1, H, 32, HD, dtype=torch.float16)
        K_out, _ = press.compress(_FakeModule(0), None, K, V, None, {})

        assert torch.equal(K_out, K)

    def test_values_quantized(self):
        from fraqtl_press import FraqtlPress

        H, HD = 4, 64
        basis = {0: _orthogonal_basis(H, HD)}
        press = FraqtlPress(bits=3, eigenbasis=basis)
        press._U_cache[0] = (
            basis[0].unsqueeze(0).float(),
            basis[0].unsqueeze(0).float().transpose(-2, -1).contiguous(),
        )

        V = torch.randn(1, H, 32, HD, dtype=torch.float16)
        K = torch.randn(1, H, 32, HD, dtype=torch.float16)
        _, V_out = press.compress(_FakeModule(0), None, K, V, None, {})

        assert not torch.equal(V_out, V)

    def test_eigenbasis_beats_random(self):
        from fraqtl_press import FraqtlPress

        torch.manual_seed(42)
        H, HD, S = 4, 64, 256

        V = torch.randn(1, H, S, HD, dtype=torch.float32)
        V[:, :, :, :4] *= 100
        V[:, :, :, 4:16] *= 10
        V = V.half()
        K = torch.randn(1, H, S, HD, dtype=torch.float16)

        # PCA basis
        Vf = V[0].float().reshape(H, S, HD)
        M = torch.bmm(Vf.transpose(-2, -1), Vf)
        _, ec = torch.linalg.eigh(M)
        pca = {0: ec.flip(-1).half()}

        # Random basis
        rand = {0: _orthogonal_basis(H, HD)}

        press_pca = FraqtlPress(bits=3, eigenbasis=pca)
        press_pca._U_cache[0] = (
            pca[0].unsqueeze(0).float(),
            pca[0].unsqueeze(0).float().transpose(-2, -1).contiguous(),
        )
        press_rand = FraqtlPress(bits=3, eigenbasis=rand)
        press_rand._U_cache[0] = (
            rand[0].unsqueeze(0).float(),
            rand[0].unsqueeze(0).float().transpose(-2, -1).contiguous(),
        )

        _, V_pca = press_pca.compress(_FakeModule(0), None, K, V, None, {})
        _, V_rand = press_rand.compress(_FakeModule(0), None, K, V, None, {})

        err_pca = (V.float() - V_pca.float()).norm()
        err_rand = (V.float() - V_rand.float()).norm()
        assert err_pca < err_rand

    def test_missing_layer_passthrough(self):
        from fraqtl_press import FraqtlPress

        press = FraqtlPress(bits=4, eigenbasis={0: _orthogonal_basis(4, 64)})
        press._U_cache[0] = (
            press.eigenbasis[0].unsqueeze(0).float(),
            press.eigenbasis[0].unsqueeze(0).float().transpose(-2, -1).contiguous(),
        )

        K = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        V = torch.randn(1, 4, 32, 64, dtype=torch.float16)
        K_out, V_out = press.compress(_FakeModule(5), None, K, V, None, {})

        assert torch.equal(K_out, K)
        assert torch.equal(V_out, V)
