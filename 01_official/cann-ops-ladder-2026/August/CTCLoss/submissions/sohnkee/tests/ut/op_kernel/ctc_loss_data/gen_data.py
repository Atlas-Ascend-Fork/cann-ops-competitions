#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import torch
import torch.nn as nn


def impl(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    """
    CTCLoss算子golden实现，直接调用torch.nn.CTCLoss

    参数:
    log_probs: (T, N, C) numpy数组，输入log概率（已沿类别维做log_softmax）
    targets: (N, S) numpy数组，目标标签序列(不含blank)
    input_lengths: (N,) numpy数组，每个batch的有效时间步数
    target_lengths: (N,) numpy数组，每个batch的有效目标序列长度
    blank: blank标签索引，默认0
    reduction: 'none' | 'mean' | 'sum'
    zero_infinity: 当正无穷时是否置零

    返回:
    neg_log_likelihood: (N,) 或 (1,) numpy数组，负对数似然损失
    """
    # 转为torch tensor
    log_probs_t = torch.from_numpy(log_probs.astype('float32'))
    targets_t = torch.from_numpy(targets.astype('int64'))
    input_lengths_t = torch.from_numpy(input_lengths.astype('int64'))
    target_lengths_t = torch.from_numpy(target_lengths.astype('int64'))

    # 直接调用PyTorch CTCLoss（输入已做log_softmax，算子内部不预处理）
    ctc_loss_fn = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)
    loss = ctc_loss_fn(log_probs_t, targets_t, input_lengths_t, target_lengths_t)

    # 转回numpy
    if reduction == 'none':
        loss_np = loss.detach().numpy().astype('float32')
    else:
        loss_np = loss.detach().numpy().reshape(1).astype('float32')

    return loss_np


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
    input_log_probs = np.ones((2, 1, 3)).astype(d_type_dict["float32"])
    input_targets = np.ones((1, 2)).astype(d_type_dict["int32"])
    input_input_lengths = np.ones((1)).astype(d_type_dict["int32"])
    input_target_lengths = np.ones((1)).astype(d_type_dict["int32"])
    attr_blank = 0
    attr_reduction = "none"
    attr_zero_infinity = False
    
    # 计算 golden 数据
    golden = impl(input_log_probs, input_targets, input_input_lengths, input_target_lengths, attr_blank, attr_reduction, attr_zero_infinity)
    
    # 保存数据到文件
    input_log_probs.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_ctc_loss_log_probs.bin")
    input_targets.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_ctc_loss_targets.bin")
    input_input_lengths.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_ctc_loss_input_lengths.bin")
    input_target_lengths.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_ctc_loss_target_lengths.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_ctc_loss.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_ctc_loss.bin")
    
    print(f"生成完成: dtype={d_type}")
