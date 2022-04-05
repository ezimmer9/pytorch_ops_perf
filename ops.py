import os


def get_ops_list():
    ops = [
        ['linear', 'input', "weight", "bias"],
        ['mul', "self1" , 'other1']
    ]

    consts = [
        {"name": "input", "shape": [64, 1024], "dtype": "kFloat32", "type": "Tensor"},
        {"name": "weight", "shape": [1024, 1024], "dtype": "kFloat32", "type": "Tensor"},
        {"name": "bias", "shape": [1024], "dtype": "kFloat32", "type": "Tensor"},
        {"name": "self1", "shape": [64], "dtype": "kFloat32", "type": "Tensor"},
        {"name": "other1", "shape": 64, "dtype": "kInt", "type": "Scalar"},
        {"name": "stride1", "shape": [2,2], "dtype": "kInt", "type": "IntArrayRef"}
        
    ]
    return ops, consts