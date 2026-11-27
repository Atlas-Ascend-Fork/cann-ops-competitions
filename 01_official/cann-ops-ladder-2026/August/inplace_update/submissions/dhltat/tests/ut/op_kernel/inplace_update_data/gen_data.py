#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import tensorflow as tf
from tensorflow.python.ops import inplace_ops

def impl(x: np.ndarray, i: np.ndarray, v: np.ndarray):
    # 拷贝避免修改外部原数组
    x_var = tf.Variable(x.copy())
    i_tf = tf.convert_to_tensor(i, dtype=tf.int32)
    # 更新值与x保持相同dtype
    v_tf = tf.convert_to_tensor(v, dtype=x_var.dtype)
    
    out_tensor = inplace_ops.inplace_update(x_var, i_tf, v_tf)
    # 转回numpy并强制和输入x同数据类型
    result_np = out_tensor.numpy().astype(x.dtype)
    return result_np


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
    input_x = np.ones((1024, 1280)).astype(d_type_dict["float32"])
    input_i = np.ones((1)).astype(d_type_dict["int32"])
    input_v = np.ones((1, 1280)).astype(d_type_dict["float32"])

    
    # 计算 golden 数据
    golden = impl(input_x, input_i, input_v)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_inplace_update_x.bin")
    input_i.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_inplace_update_i.bin")
    input_v.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_inplace_update_v.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float32_golden_inplace_update.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float32"]).tofile("float32_golden_inplace_update.bin")
    
    print(f"生成完成: dtype={d_type}")
