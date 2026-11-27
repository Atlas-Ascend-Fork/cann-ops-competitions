#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch
import torch.nn.functional as F


def impl(x, gamma, beta, num_groups, epsilon=1e-5):
    """
    GroupNorm算子golden实现，直接调用F.group_norm

    参数:
    x: (N, C, *) numpy数组，输入张量（原始值，无预处理）
    gamma: (C,) numpy数组，逐通道缩放系数
    beta: (C,) numpy数组，逐通道偏置
    num_groups: int，分组数
    epsilon: float，数值稳定常数，默认1e-5

    返回:
    y: 与x同shape同dtype的numpy数组
    """
    orig_dtype = x.dtype
    x_t = torch.from_numpy(x.astype('float32'))
    gamma_t = torch.from_numpy(gamma.astype('float32'))
    beta_t = torch.from_numpy(beta.astype('float32'))

    y_t = F.group_norm(x_t, num_groups, gamma_t, beta_t, eps=epsilon)

    if orig_dtype == np.float16:
        y_np = y_t.half().numpy()
    elif orig_dtype == np.float32:
        y_np = y_t.float().numpy()
    else:
        y_np = y_t.numpy().astype(orig_dtype)

    return y_np


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float32"
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
    input_x = np.ones((1, 4, 2, 2)).astype(d_type_dict["float32"])
    input_gamma = np.ones((4)).astype(d_type_dict["float32"])
    input_beta = np.ones((4)).astype(d_type_dict["float32"])
    attr_num_groups = 2
    attr_epsilon = 1e-05
    
    # 计算 golden 数据
    golden = impl(input_x, input_gamma, input_beta, attr_num_groups, attr_epsilon)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_group_norm_x.bin")
    input_gamma.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_group_norm_gamma.bin")
    input_beta.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_group_norm_beta.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_group_norm.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_group_norm.bin")
    
    print(f"生成完成: dtype={d_type}")
