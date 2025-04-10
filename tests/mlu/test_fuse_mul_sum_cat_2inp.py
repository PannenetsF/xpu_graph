import pytest
import math

import torch
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

def fn0(x1, x2, x3, x4):
    a = x1 * x2
    sum_a = a.sum(dim=1)
    b = x3 * x4
    sum_b = b.sum(dim=1)
    out = torch.cat([sum_a, sum_b], dim=1)
    return out

def mul_sum_cat_test(xpu_graph_backend, func):
    batch = 86
    dtype = torch.half
    a = torch.randn(batch, 80, 32, dtype=dtype, device="mlu")
    b = torch.randn(batch, 80, 32, dtype=dtype, device="mlu")
    c = torch.randn(1, 80, 32, dtype=dtype, device="mlu")
    d = torch.randn(batch, 80, 32, dtype=dtype, device="mlu")

    res1 = func(a, b, c, d)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res = compiled(a, b, c, d)
    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.001, use_MSE=True, use_RAE=True
    )

class TestMulSumCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_mul_sum_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            mul_sum_cat_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedMulSumCat changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
    )
    mul_sum_cat_test(xpu_graph_backend, fn0)
