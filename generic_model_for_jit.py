import os
import sys
import argparse
import torch
import intel_extension_for_pytorch as ipex

__doc__ = "Generate JIT model for linear to be able to run with IPEX"
parser = argparse.ArgumentParser('Run-Linear', description=__doc__)
parser.add_argument('-K', '--K_DIM', type=int, required=False, default=128, dest="K", help="K DIM")
parser.add_argument('-M', '--M_DIM', type=int, required=False, default=128, dest="M", help="M DIM")

ops_with_params = ['linear', 'conv2d']

class Model_linear(torch.nn.Module):
    def __init__(self, data_type, params):
        super(Model_linear, self).__init__()
        self.linear = torch.nn.Linear(params[1], params[0], dtype=data_type)
    def forward(self, input):
        return self.linear(input)

class Model_mm(torch.nn.Module):
    def __init__(self, data_type, params):
        super(Model_mm, self).__init__()
    def forward(self, input1, input2):
        return torch.mm(input1, input2)

class Model_matmul(torch.nn.Module):
    def __init__(self, data_type, params):
        super(Model_matmul, self).__init__()
    def forward(self, input1, input2):
        return torch.matmul(input1, input2)

class Model_bmm(torch.nn.Module):
    def __init__(self, data_type, params):
        super(Model_bmm, self).__init__()
    def forward(self, input1, input2):
        return torch.bmm(input1, input2)

class Model_conv2d(torch.nn.Module):
    def __init__(self, data_type, params):
        super(Model_conv2d, self).__init__()
        self.conv2d = torch.nn.Conv2d(params[0], params[1],params[2],params[3],params[4],params[5],params[6], bias=True, padding_mode='zeros', device=None, dtype=data_type)
    def forward(self, input):
        return self.conv2d(input)
# Tensor conv2d(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups); // {"schema": "aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor", "dispatch": "False", "default": "True"}


def generic_model_for_jit(model_class, data_type, params):
    model=model_class(data_type, params)
    model.eval()
    model_jit=torch.jit.script(model)
    model_file_name = model_class.__name__+'_jit.pt'
    model_jit.save(model_file_name)
    return model_jit
def parse_params(ops, consts):
    op_name=ops[0]
    cpp_dtype= consts[0]['dtype']
    if cpp_dtype=='kBFloat16':
        data_type=torch.bfloat16
    elif cpp_dtype=='kInt8':
        data_type=torch.int8
    elif cpp_dtype=='kFloat32':
        data_type=torch.float32
    else:
        print("non supported dtype in IPEX OP", cpp_dtype)
        return 0
    if op_name=='linear':
        params=consts[1]['shape']
    elif op_name in ['mm', 'matmul', 'bmm']:
        params=[]
    elif op_name=='conv2d':
        params=[]
        params.append(consts[1]['shape'][1])        
        params.append(consts[1]['shape'][0])        
        params.append(consts[1]['shape'][2])        
        params.append(consts[3]['shape'])
        params.append(consts[4]['shape'])
        params.append(consts[5]['shape'])
        params.append(consts[6]['shape'])
    return params, data_type 

def parse_input_params(ops, consts):
    op_name=ops[0]
    if op_name=='linear':
        input_params =consts[0]['shape']
    elif op_name=='mm':
        input_params=[consts[0]['shape'][0],consts[0]['shape'][1],consts[1]['shape'][1]]
    elif op_name in ['matmul', 'bmm'] :
        input_params=[]
        input_params.append(len(consts[0]['shape']))
        input_params.append(len(consts[1]['shape']))
        for shape in consts[0]['shape']:
            input_params.append(shape)
        for shape in consts[1]['shape']:
            input_params.append(shape)
    elif op_name=='conv2d':
        input_params=[]
        input_params.append(len(consts[0]['shape']))
        for shape in consts[0]['shape']:
            input_params.append(shape)
    return input_params 

def prepare_op_model_and_inputs(ops, consts):
    Model_for_jit=getattr(sys.modules[__name__], 'Model_'+str(ops[0]))
    consts_for_op=[]
    for c in consts:
        for op_in in ops[1:]:
            if c['name']==op_in:
                consts_for_op.append(c)
                break
    if ops[0] in ops_with_params or (not os.path.exists(Model_for_jit.__name__+'_jit.pt')):
        params, data_type = parse_params(ops, consts_for_op)
        model_jit = generic_model_for_jit(Model_for_jit, data_type, params)
    input_params = parse_input_params(ops, consts_for_op)
    return input_params
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:]) 
    ##### Linear
    Model_for_jit = Model_linear
    params = [args.K, args.M]
    data_type=torch.bfloat16
    model_jit = generic_model_for_jit(Model_for_jit, data_type, params)
    out=model_jit(torch.ones((1,args.M), dtype=torch.bfloat16))
    print(out.shape)
    ##### mm
    Model_for_jit = Model_mm
    params = []
    data_type=torch.bfloat16
    model_jit = generic_model_for_jit(Model_for_jit, data_type, params)
    input_1=torch.ones((64,128), dtype=torch.bfloat16)
    input_2=torch.ones((128,256), dtype=torch.bfloat16)
    out=model_jit(input_1,input_2)
    print(out.shape)    
    ##### matmul
    Model_for_jit = Model_matmul
    params = []
    data_type=torch.bfloat16
    model_jit = generic_model_for_jit(Model_for_jit, data_type, params)
    input_1=torch.ones((64,128), dtype=torch.bfloat16)
    input_2=torch.ones((128,256), dtype=torch.bfloat16)
    out=model_jit(input_1,input_2)
    print(out.shape)
    ##### bmm
    Model_for_jit = Model_bmm
    params = []
    data_type=torch.bfloat16
    model_jit = generic_model_for_jit(Model_for_jit, data_type, params)
    input_1=torch.ones((33,64,128), dtype=torch.bfloat16)
    input_2=torch.ones((33,128,256), dtype=torch.bfloat16)
    out=model_jit(input_1,input_2)
    print(out.shape)    
   ##### Conv2d
    Model_for_jit = Model_conv2d
    params = [1,32,3, [1,1], [0,0],[1,1],1]
    data_type=torch.bfloat16
    model_jit = generic_model_for_jit(Model_for_jit, data_type, params)
    input_conv=torch.ones((64,1,28,28), dtype=torch.bfloat16)
    out=model_jit(input_conv)
    print(out.shape)    

