import pytest

import torch
import torch_mlu
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
aten = torch.ops.aten


def fn0(x):
    axis = -1
    c = torch.cat(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        axis,
    )
    return (c,)


def fn1(x):
    axis = -1
    c = torch.cat(
        [
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
            x[:, 39415:39479],
            x[:, 40264:40328],
            x[:, 24347:24411],
            x[:, 23854:23918],
            x[:, 23361:23425],
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
        ],
        axis,
    )
    c = c + 1
    slice_190 = x[:, 35263:35295]
    slice_192 = x[:, 35296:35328]
    slice_194 = x[:, 35329:35361]
    slice_196 = x[:, 35378:35410]
    slice_198 = x[:, 35411:35443]
    slice_200 = x[:, 35444:35476]
    slice_202 = x[:, 35971:36003]
    slice_204 = x[:, 37206:37238]
    slice_206 = x[:, 37256:37288]
    slice_208 = x[:, 37306:37338]
    slice_210 = x[:, 37391:37423]
    slice_212 = x[:, 37432:37464]
    slice_214 = x[:, 37474:37506]
    slice_216 = x[:, 1165:1197]
    slice_218 = x[:, 1340:1372]
    slice_220 = x[:, 1387:1419]
    slice_222 = x[:, 1739:1771]
    slice_224 = x[:, 2629:2661]
    slice_226 = x[:, 2676:2708]
    slice_228 = x[:, 2723:2755]
    slice_230 = x[:, 2897:2929]
    slice_232 = x[:, 2186:2218]
    slice_234 = x[:, 2232:2264]
    slice_236 = x[:, 2278:2310]
    slice_238 = x[:, 2417:2449]

    stack_4 = torch.cat(
        [
            slice_190,
            slice_192,
            slice_194,
            slice_196,
            slice_198,
            slice_200,
            slice_202,
            slice_204,
            slice_206,
            slice_208,
            slice_210,
            slice_212,
            slice_214,
            slice_216,
            slice_218,
            slice_220,
            slice_222,
            slice_224,
            slice_226,
            slice_228,
            slice_230,
            slice_232,
            slice_234,
            slice_236,
            slice_238,
        ],
        dim=-1,
    )
    return (torch.cat([stack_4, c], dim=-1),)


def fn5(x):
    axis = -1
    b = x.clone()
    c = torch.cat(
        [
            b,
            x[:, 34691:34755],
            x[:, 38371:38435],
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
        ],
        axis,
    )
    return (c, 1)


def fn6(x):
    b = x.clone()
    d = b.clone()
    c = aten.cat(
        [
            b,
            x[:, 34691:34755],
            d[:, 38371:38435],
            d[:, 38371:38437],
        ],
        1,
    )
    return (c, 1)


def fn7(x):
    b = x.clone()
    d = b.clone()
    c = aten.cat(
        [
            b,
            x[:, 34691:34755],
            x[:, 38371:38435],
            d,
            x[:, 41479:41543],
            x[:, 33535:33599],
            x[:, 34065:34129],
        ],
        1,
    )
    return (c, 1)


def fn9(x):
    c = aten.cat(
        [
            x[:, 43106 - 2 : 9223372036854775807],
            x[:, 43106 - 2 : 9223372036854775807],
        ],
        1,
    )
    return (c, 1)


def fn10(x):
    c = aten.cat(
        [
            x[:, 43106 - 2 :],
            x[:, 43106 - 2 :],
        ],
        1,
    )
    return (c, 1)


def fn11(x):
    c = aten.cat(
        [
            x[:, 43106 - 2 : -1],
            x[:, 43106 - 2 : -1],
        ],
        1,
    )
    return (c, 1)


def fn12(x):
    x1 = x[:, 34691:34755] + 1
    x2 = x[:, 38371:38435] + 1
    x3 = x[:, 41479:41543] + 1
    x4 = x[:, 33535:33599] + 1
    x5 = x[:, 34065:34129] + 1
    x6 = x[:, 39415:39479] + 1
    x7 = x[:, 40264:40328] + 1
    x8 = x[:, 24347:24411] + 1
    x9 = x[:, 23854:23918] + 1
    x10 = x[:, 23361:23425] + 1
    x11 = x[:, 34691:34755] + 1
    x12 = x[:, 38371:38435] + 1
    return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12


def fn13(x):
    x1 = x[:, 0:64] + 1
    x2 = x[:, 64:128] + 1
    x3 = x[:, 128:192] + 1
    x4 = x[:, 192:256] + 1
    x5 = x[:, 256:384] + 1
    x6 = x[:, 384:512] + 1
    x7 = x[:, 512:640] + 1
    return x1, x2, x3, x4, x5, x6, x7


def fn14(x):
    tmp_Test = [
        12573,
        11984,
        11562,
        9782,
        9727,
        9745,
        9763,
        9773,
        9805,
        9822,
        10283,
        10736,
    ]
    input_list = []
    for index in tmp_Test:
        input_list.append(x[:, index : index + 4] + 1)
    return input_list


def fn15(x):
    tmp_Test = [
        9425,
        18012,
        11675,
        12033,
        9926,
        10387,
        10840,
        11253,
        9457,
        18044,
        11707,
        12065,
        9958,
        10419,
        10872,
        11285,
        9489,
        18076,
        11739,
        12097,
        9990,
        10451,
        10904,
        11317,
        9521,
        18108,
        11771,
        12129,
        10022,
        10483,
        10936,
        11349,
        9553,
        18140,
        11803,
        12161,
        10054,
        10515,
        10968,
        11381,
        9585,
        18172,
        11835,
        12193,
        10086,
        10547,
        11000,
        11413,
        9617,
        18204,
        11867,
        12225,
        10118,
        10579,
        11032,
        11445,
    ]
    input_list = []
    for index in tmp_Test:
        input_list.append(x[:, index : index + 32] + 1)
    return input_list


def slice_test(xpu_graph_backend, func):
    for batch in (10, 512, 64, 31):
        a = torch.randn(batch, 43106).to(device=device)
        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res = compiled(a)[0]
        res1 = func(a)[0]
        assert torch.equal(res.cpu().float(), res1.cpu().float())


class TestCatSlice:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level1
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn5,
            fn6,
            fn7,
            fn9,
            fn10,
            fn11,
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            slice_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedCatSlice changed graph" in caplog.text


class TestSlice:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level1
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn12,
            fn13,
            fn14,
            fn15,
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            slice_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedSlice changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, freeze=True, opt_level=OptLevel.level1,
    )
    slice_test(xpu_graph_backend, fn0)
