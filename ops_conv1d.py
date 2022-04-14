import os


def get_ops_list():
    ops = [
        ['conv1d', "input" , "weight"]

#- func: conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor

    ]

    consts = [
        {"name": "input", "shape": [2619, 1437, 1], "dtype": "kFloat32", "type": "Tensor"},
        {"name": "weight", "shape": [2619, 1437, 1], "dtype": "kFloat32", "type": "Tensor"},
        {"name": "bias", "shape": [2619], "dtype": "kFloat32", "type": "Tensor"},
        {"name": "stride", "shape": 1, "dtype": "kint8", "type": "IntArrayRef"},
        {"name": "padding", "shape": 0, "dtype": "kInt8", "type": "IntArrayRef"},
        {"name": "dilation", "shape":1, "dtype": "kInt8", "type": "IntArrayRef"},
        {"name": "groups", "shape": 1, "dtype": "kInt64", "type": "int64_t"}
        
    ]
    return ops, consts