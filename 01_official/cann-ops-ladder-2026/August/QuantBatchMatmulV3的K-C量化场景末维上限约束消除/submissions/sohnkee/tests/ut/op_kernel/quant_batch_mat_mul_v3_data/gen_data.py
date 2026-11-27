#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def _bfloat16_dtype():
    try:
        from ml_dtypes import bfloat16
        return bfloat16
    except ImportError:
        pass

    try:
        import tensorflow as tf
        return tf.bfloat16.as_numpy_dtype
    except ImportError:
        return None


def _float32_to_bfloat16_bits(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def _to_bfloat16(value):
    dtype = _bfloat16_dtype()
    if dtype is not None:
        return np.asarray(value, dtype=np.float32).astype(dtype)
    return _float32_to_bfloat16_bits(value)


def impl(x1, x2, scale, pertokenScale, transposeX1, transposeX2):
    x1_i32 = np.asarray(x1, dtype=np.int32)
    x2_i32 = np.asarray(x2, dtype=np.int32)
    if transposeX1:
        x1_i32 = np.swapaxes(x1_i32, -1, -2)
    if transposeX2:
        x2_i32 = np.swapaxes(x2_i32, -1, -2)

    scale_f32 = np.asarray(scale, dtype=np.float32).reshape(1, -1)
    pertoken_scale_f32 = np.asarray(pertokenScale, dtype=np.float32).reshape(-1, 1)

    matmul_res = np.matmul(x1_i32, x2_i32).astype(np.float32)
    out = matmul_res * scale_f32 * pertoken_scale_f32
    return _to_bfloat16(out)


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "int8"
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
    input_x1 = np.ones((128, 65536)).astype(d_type_dict["int8"])
    input_x2 = np.ones((65536, 96)).astype(d_type_dict["int8"])
    input_scale = np.ones((96)).astype(d_type_dict["float32"])
    input_pertokenScale = np.ones((128)).astype(d_type_dict["float32"])
    attr_transposeX1 = False
    attr_transposeX2 = False
    
    # 计算 golden 数据
    golden = impl(input_x1, input_x2, input_scale, input_pertokenScale, attr_transposeX1, attr_transposeX2)
    
    # 保存数据到文件
    input_x1.astype(d_type_dict["int8"]).tofile(f"{d_type}_input_quant_batch_mat_mul_v3_x1.bin")
    input_x2.astype(d_type_dict["int8"]).tofile(f"{d_type}_input_quant_batch_mat_mul_v3_x2.bin")
    input_scale.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_quant_batch_mat_mul_v3_scale.bin")
    input_pertokenScale.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_quant_batch_mat_mul_v3_pertokenScale.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("bfloat16_golden_quant_batch_mat_mul_v3.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["bfloat16"]).tofile(_f)
        else:
            golden.astype(d_type_dict["bfloat16"]).tofile("bfloat16_golden_quant_batch_mat_mul_v3.bin")
    
    print(f"生成完成: dtype={d_type}")
