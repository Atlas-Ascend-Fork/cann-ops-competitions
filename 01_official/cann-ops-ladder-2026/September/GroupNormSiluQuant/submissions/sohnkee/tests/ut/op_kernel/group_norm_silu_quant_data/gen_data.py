#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(self, gamma, beta, quantScale, group, eps=1e-5, activateSilu=True):
    """GroupNormSiluQuant算子golden实现

    参数名与顺序与 JSON 的 input_desc + attr_desc 一致：
    self: (N, C, ...) numpy数组，输入张量
    gamma: (C,) numpy数组，缩放
    beta: (C,) numpy数组，偏置
    quantScale: 标量 或 (C,) numpy数组，量化尺度
    group: int，分组数
    eps: float，数值稳定常数，默认 1e-5
    activateSilu: bool，是否开启 SiLU，默认 true

    返回:
    out: 同 self shape 的 int8 量化结果
    meanOut: (N, group) numpy数组，均值
    rstdOut: (N, group) numpy数组，标准差倒数
    """
    orig_dtype = self.dtype
    N, C = self.shape[0], self.shape[1]
    Cpg = C // group
    spatial = self.shape[2:]

    x32 = self.astype(np.float32)
    gamma32 = gamma.astype(np.float32)
    beta32 = beta.astype(np.float32)
    qs32 = quantScale.astype(np.float32)

    # GroupNorm: reshape 到 (N, group, Cpg, ...)
    x_g = x32.reshape(N, group, Cpg, *spatial)
    axes = tuple(range(2, 2 + len(spatial) + 1))        # (Cpg, *spatial) 归约维
    mean = x_g.mean(axis=axes, keepdims=True)           # (N, group, 1, ...)
    var = x_g.var(axis=axes, keepdims=True)             # (N, group, 1, ...)
    rstd = 1.0 / np.sqrt(var + eps)                     # (N, group, 1, ...)

    gamma_r = gamma32.reshape(1, group, Cpg, *([1] * len(spatial)))
    beta_r = beta32.reshape(1, group, Cpg, *([1] * len(spatial)))

    gn_out = (x_g - mean) * rstd * gamma_r + beta_r

    # SiLU
    if activateSilu:
        silu_out = gn_out / (1.0 + np.exp(-gn_out))
    else:
        silu_out = gn_out

    # Quant: round + clip 到 int8
    if qs32.size == 1:
        qs_r = qs32
    else:
        qs_r = qs32.reshape(1, group, Cpg, *([1] * len(spatial)))
    quant_out = np.clip(np.rint(silu_out / qs_r), -128, 127).astype(np.int8)

    out = quant_out.reshape(N, C, *spatial)

    mean_np = mean.reshape(N, group)
    rstd_np = rstd.reshape(N, group)
    if orig_dtype == np.float16:
        mean_np = mean_np.astype(np.float16)
        rstd_np = rstd_np.astype(np.float16)
    else:
        mean_np = mean_np.astype(np.float32)
        rstd_np = rstd_np.astype(np.float32)

    return out, mean_np, rstd_np


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float16"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]
    
    # 生成输入数据
    input_self = np.ones((1, 4, 2, 2)).astype(d_type_dict["float16"])
    input_gamma = np.ones((4)).astype(d_type_dict["float16"])
    input_beta = np.ones((4)).astype(d_type_dict["float16"])
    input_quantScale = np.ones((1)).astype(d_type_dict["float32"])
    attr_group = 2
    attr_eps = 1e-05
    attr_activateSilu = True
    
    # 计算 golden 数据
    golden = impl(input_self, input_gamma, input_beta, input_quantScale, attr_group, attr_eps, attr_activateSilu)
    
    # 保存数据到文件
    input_self.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_group_norm_silu_quant_self.bin")
    input_gamma.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_group_norm_silu_quant_gamma.bin")
    input_beta.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_group_norm_silu_quant_beta.bin")
    input_quantScale.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_group_norm_silu_quant_quantScale.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            _out_dtypes = ["int8", "float16", "float16"]
            for _gi, _g in enumerate(golden):
                _dt = _out_dtypes[_gi] if _gi < len(_out_dtypes) else _out_dtypes[-1]
                _g.astype(d_type_dict[_dt]).tofile(f"{_dt}_golden_group_norm_silu_quant_{_gi}.bin")
        else:
            golden.astype(d_type_dict["int8"]).tofile("int8_golden_group_norm_silu_quant_0.bin")
    
    print(f"生成完成: dtype={d_type}")
