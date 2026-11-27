#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import tensorflow as tf

def impl(labels, predictions, weights, num_classes, dtype):
    if weights is not None and isinstance(weights, (list, np.ndarray)) and len(weights) == 0:
        weights = None
    if num_classes == -1:
        num_classes = None    
    if dtype == 'int32':
        tf_dtype = tf.int32
    elif dtype == 'float32':
        tf_dtype = tf.float32
    else:
        tf_dtype = tf.int32
    
    cm_tensor = tf.math.confusion_matrix(
        labels,
        predictions,
        num_classes=num_classes,
        weights=weights,
        dtype=tf_dtype
    )
    return cm_tensor.numpy()


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "int32"
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
    input_labels = np.ones((5)).astype(d_type_dict["int32"])
    input_predictions = np.ones((5)).astype(d_type_dict["int32"])
    input_weights = None  # 可选输入，未提供
    attr_num_classes = 3
    attr_dtype = "int32"
    
    # 计算 golden 数据
    golden = impl(input_labels, input_predictions, input_weights, attr_num_classes, attr_dtype)
    
    # 保存数据到文件
    input_labels.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_confusion_matrix_labels.bin")
    input_predictions.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_confusion_matrix_predictions.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("int32_golden_confusion_matrix.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["int32"]).tofile(_f)
        else:
            golden.astype(d_type_dict["int32"]).tofile("int32_golden_confusion_matrix.bin")
    
    print(f"生成完成: dtype={d_type}")
