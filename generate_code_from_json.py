import os
import sys
import subprocess
from typing import final

from numpy import double
import code_gen
import argparse
import shutil
import torch
#from ops import get_ops_list
import json

parser = argparse.ArgumentParser('Generate-Code-From-JSON', description=__doc__)
parser.add_argument('-ppets', '--ppet_sweep', type=str, required=False, default=None,
                    dest='ppet_output_file', help="Run sweep with input file from ppet, which dump all the OPs not yet in compute ML model")

# https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/types.h
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/python_scalars.h
# ../libraries.ai.profiling.pt-sampler/ptsampler/network.py
Tensor_type_dict = {'INT8': 'kInt8',
                    'UINT8': 'kUInt8',
                    'FP16': 'kFloat16',
                    'BF16': 'kBFloat16',
                    'INT16': 'kInt16',
                    'UINT16': 'kInt16',
                    'INT32': 'kInt32',
                    'UINT32': 'kInt32',
                    'INT64': 'kInt64',
                    'UINT64': 'kInt64',
                    'FLOAT': 'kFloat32',
                    'FP32': 'kFloat32',
                    'FP64': 'kFloat64',
                    'FLOAT64': 'kFloat64',
                    'BIN': 'kBool',
                        }
args_type_dict = { list: ['IntArrayRef', '::std::array<bool,2>', '::std::array<bool,3>', '::std::array<bool,4>'] ,
                   int: ['Scalar', 'int64_t', 'ScalarType'] ,  #'ScalarType'
                   bool: ['bool'],
                   str: ['Dimname', 'c10::string_view', 'Layout', 'Device', 'MemoryFormat'],
                   float: ['double', 'Scalar']
                        }
arg_types_to_pytorch_types = { 'IntArrayRef': "kint32",
                                'ScalarType': "ScalarType::Float",
                                'Scalar': "kInt32",
                                'int64_t': "kInt64",
                                'bool': "kBool",
                                'double': 'kFloat64',
                                '::std::array<bool,2>': 'kBool',
                                '::std::array<bool,3>': 'kBool',
                                '::std::array<bool,4>': 'kBool',
                                'Layout': 'at::kStrided',
                                'Device': 'DeviceType::CPU',
                                'MemoryFormat': 'MemoryFormat::Preserve'
}

def read_from_json(filename):
    with open(filename, "r") as f:
        json_dict = json.load(f)
    #print("Reading file: ",filename) 
    return json_dict

def new_signature(input_list):
    input_struct = []
    for in_iter in input_list:
          input_struct.append(in_iter.split(' '))
    return input_struct

def parse_sig_file(filename):
    with open(filename, "r") as f:
        all_lines  = f.readlines()
    PT_out_formats = ('Tensor ', 'void ', 'bool ', 'int64_t ', '::std::tuple<Tensor &,Tensor &> ','::std::tuple<Tensor,Tensor> ', '::std::vector<Tensor> ', 
                   '::std::tuple<Tensor,Tensor,Tensor> ', '::std::tuple<Tensor &,Tensor &,Tensor &> ','::std::tuple<Tensor,Tensor,Tensor,Tensor> ',
                   '::std::tuple<Tensor &,Tensor &,Tensor &,Tensor &> ', '::std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> ', '::std::tuple<double,int64_t> ',
                   '::std::tuple<Tensor,Tensor,Tensor,::std::vector<Tensor>> ', '::std::tuple<Tensor,Tensor,double,int64_t> ',
                   '::std::tuple<Tensor,Tensor,Tensor,Tensor,int64_t> ',
                    'const Tensor ', 'Scalar ', 'ScalarType ', 'QScheme ', 'double ')
    OP_signature_dict={}
    num_ops=0
    for op_sig_line in all_lines:
        if op_sig_line.startswith(PT_out_formats):
            start_str = ''.join([c for c in PT_out_formats if op_sig_line.startswith(c)])
            line_without_out_format = op_sig_line.replace(start_str,'',1)
            if line_without_out_format.startswith('& '):
                line_without_out_format=line_without_out_format[2:]
            line_split_1 = line_without_out_format.split('(', 1)
            line_split_2 = line_split_1[1].split(')', 1)
            op_name = line_split_1[0]
            input_list = line_split_2[0].split(', ')
            #print(op_name)
            #print(input_list)
            if op_name in OP_signature_dict:
                OP_signature_dict[op_name]['num_sig']+=1
                in_list_struct = new_signature(input_list)
                OP_signature_dict[op_name]['signatures'].append(in_list_struct)
            else:
                OP_signature_dict[op_name]={}
                OP_signature_dict[op_name]['num_sig']=1
                in_list_struct=new_signature(input_list)
                OP_signature_dict[op_name]['signatures']=[in_list_struct]

            num_ops+=1
    #print(OP_signature_dict)
    #print("num OPs=",num_ops)
    return OP_signature_dict
def count_tensors_in_sig(sig):
    tensors_in_sig_all = 0
    tensors_in_sig_optional=0
    self_tensor_location = -1
    tensor_num=-1
    for i, ins in enumerate(sig):
        if any('Tensor' in k for k in ins):
            tensors_in_sig_all+=1
            tensor_num+=1
        if any('self' in k for k in ins):
            self_tensor_location=tensor_num
        if any('optional' in k for k in ins):
            tensors_in_sig_optional+=1
    tensors_in_sig_must=tensors_in_sig_all-tensors_in_sig_optional
    return tensors_in_sig_must, tensors_in_sig_all, self_tensor_location
def check_all_args_exist_in_sig(sig,all_args):
    all_args_in_sig = True
    arg_types=[]
    for args in all_args:
        arg_type=args_type_dict[type(args)]
        this_arg_in_sig=False
        for ins in sig:
            for r in arg_type:
                if any(r in k for k in ins):
                    this_arg_in_sig=True
                    arg_types.append(r)
        all_args_in_sig = all_args_in_sig and this_arg_in_sig
    return all_args_in_sig, arg_types
                

def tensor_dtype_dict(tensor_type_json):
    tensor_dtype = Tensor_type_dict[tensor_type_json]
    return tensor_dtype
def find_possible_signatures(num_tensors_in, all_args, op_signatures):
    #TODO: Build the dictionary for args
    possible_signatures = []
    arg_types=[]
    self_tensor_location=[]
    for sig in op_signatures['signatures']:
        # Rule #1: find signatures with same number of tensors
        tensors_in_sig_must, tensors_in_sig_all, self_tensor_location_tmp = count_tensors_in_sig(sig)
        num_inputs = num_tensors_in+len(all_args)
        all_args_in_sig, arg_types_tmp = check_all_args_exist_in_sig(sig,all_args)
        if num_tensors_in>=tensors_in_sig_must and num_tensors_in<=tensors_in_sig_all and all_args_in_sig and num_inputs<=len(sig):
            possible_signatures.append(sig)
            arg_types.append(arg_types_tmp)
            self_tensor_location.append(self_tensor_location_tmp)
            # print(sig)
    return possible_signatures, arg_types, self_tensor_location    

def generate_ops_from_json(args):
    try:
        print("\nConda enviroments is: {}\n".format(os.environ["CONDA_DEFAULT_ENV"]))
    except:
        print("\n----- you are not in conda env - make sure pytorch is install -----\n")

    sig_file_name = 'RegistrationDeclarations.h'
    signature_ops = parse_sig_file(sig_file_name)
    json_dict = read_from_json(args.ppet_output_file)
    json_ops = {}
    ops=[]
    consts=[]
    for op_count, ops_in_json in enumerate(json_dict['layers']):
        if '::' in ops_in_json['optype']:
            op_type = ops_in_json['optype'].split('::',1)[1]
        else:
            print("Warning, Do not support non :: OPs : ", ops_in_json['optype'])
            continue
        op_signatures = signature_ops[op_type]
        json_ops[op_type]={}
        json_ops[op_type]['Tensors']=[]
        json_ops[op_type]['args']=[]
        ops_input_list=[op_type]
        num_tensors_in=len(ops_in_json['inputs'])
        possible_signatures, arg_types, self_tensor_location = find_possible_signatures(num_tensors_in, ops_in_json['args'], op_signatures)
        if len(possible_signatures) >1:
            print("Warning, number of signatures > 1 in", op_type, ". Taking the first one. num_signatures= ", len(possible_signatures))
        if len(possible_signatures)==0:
            print(f"Warning, Didn't find any signature for {op_type}. Skiping OP")
            continue
        for i, tensor_in in enumerate(ops_in_json['inputs']):
            tensor_info = json_dict['tensors'][tensor_in]
            tensor_name = tensor_info['name'].replace('-','_')
            tensor_dtype = tensor_dtype_dict(tensor_info['dtype'])
            tensor_shape = tensor_info['shape']
            json_ops[op_type]['Tensors'].append(tensor_info)
            if i == self_tensor_location[0]:
                ops_input_list.append(tensor_name+' self')
            else:
                ops_input_list.append(tensor_name)
            if not any(c['name']==tensor_name for c in consts):
                consts.append({"name": tensor_name, "shape": tensor_shape, "dtype": tensor_dtype, "type": "Tensor"})
        for j, args_in in enumerate(ops_in_json['args']):
            arg_name = op_type + '_' + str(op_count) + '_args_' + str(j+1)
            if type(args_in) is bool:
                args_in=str(args_in).lower()
            json_ops[op_type]['args'].append(args_in)
            ops_input_list.append(arg_name)
            consts.append({"name": arg_name, "shape": args_in, "dtype": arg_types_to_pytorch_types[arg_types[0][j]], "type": arg_types[0][j]})
        ops.append(ops_input_list)    
    
    #print(json_ops)
    return ops, consts
 

if __name__ == "__main__":
    args = parser.parse_args()
    generate_ops_from_json(args)
