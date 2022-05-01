import os
import copy

def add_start_sample():
    return "   couners_events.sample(start_perf_results);\n"

def add_end_sample():
    s = "   couners_events.sample(end_perf_results);\n" + \
        "   res = get_diff_counters(start_perf_results , end_perf_results);\n" + \
        "   results.push_back(res);\n\n"
    return s
def join_op_inputs(op, consts):
    op_inputs_list=[]
    for a in op:
        for b in consts:
            if a==b['name']:
                if b['type']=='Tensor':
                    op_inputs_list.append(a)
                    break
                else:
                    if type(b["shape"]) is list:
                        if len(b["shape"])==0:
                            shape="0"
                        else:
                            str_list=[]
                            for int_ in b["shape"]:
                                str_list.append(str(int_).lower())
                            #shape = "{" + ", ".join(str_list) + "}"
                            shape = "(" + ", ".join(str_list) + ")"
                    else:
                        shape = str(b['shape'])
                    op_inputs_list.append(shape)
                    break
    assert len(op_inputs_list)==len(op), "Error, didnt find all inputs in consts list" 
    op_inputs = ', '.join(op_inputs_list)
    return op_inputs


def main(ops , consts):
    index = 0
    cpp_code = "#include <iostream>\n" + \
                "#include <vector>\n" + \
                "#include <stdlib.h>\n" + \
                "#include <time.h>\n" + \
                "#include <unistd.h>\n" + \
                "#include <torch/torch.h>\n" + \
                "#include <ATen/ATen.h>\n\n" + \
                '#include "perf_events.hpp"\n' + \
                '#include "common.h"\n\n'

    cpp_code += "std::vector<PerfResults> results;\n\n"

    cpp_code += "using namespace at;\n\n"
    
    cpp_code += "int main() {\n"
    cpp_code += "   CounterList counters;\n"
    cpp_code += "   counters = get_defaults_counters();\n"
    cpp_code += "   PerfEventsCounter couners_events(counters);\n"
    cpp_code += "   couners_events.enable();\n"
    cpp_code += "   PerfResults start_perf_results, end_perf_results, res;\n"

    for inp in consts:
        if inp["type"] == "Tensor":
            cpp_code += "   {} {};\n".format(inp["type"], inp["name"])
            if isinstance(inp['shape'], list):
                str_list = [str(int_) for int_ in inp["shape"]]
                shape = "{" + ", ".join(str_list) + "}"
                if inp["type"] == "Tensor":
                    cpp_code += "   {} = torch::ones({} , TensorOptions(kCPU).dtype(torch::{})); \n".format(inp["name"], shape, inp["dtype"])
                #elif inp["type"] == "IntArrayRef":
                #    cpp_code += "   {} = {};\n".format(inp["name"], shape)
            else:
                cpp_code += "   {} = {};\n".format(inp["name"], inp["shape"])
            
    
    cpp_code += "\n"
    ops_copy = copy.deepcopy(ops)
    for op in ops_copy:
        op_func = op.pop(0)
        if any(' self' in el for el in op):
            op = [i.replace(' self','') for i in op]
            op_str = op_func + "("    # + ", ".join(op) + ")"
            # self_input =[k for k in op if ' self' in k]
            # op.pop(op.index(self_input[0]))
            # self_input=self_input[0].split(' ',1)[0]
            # op_str = self_input + "." + op_func + "("  # + ", ".join(op) + ")"
        else:
            op_str = op_func + "("    # + ", ".join(op) + ")"
        op_inputs = join_op_inputs(op, consts)
        op_str = op_str + op_inputs +")"
        '''add the start sample'''
        cpp_code += add_start_sample()
        cpp_code += "   Tensor out{} = {};\n".format(index, op_str)
        '''add the end sample + get diff between counter + push to vector'''
        cpp_code += add_end_sample()
        index += 1

    cpp_code += "   dump_results_to_file(results);\n\n"
    cpp_code += '   std::cout << "Done" << std::endl;\n'

    cpp_code += "}"

    with open("cpp_gen.cpp", 'w') as write_file:
        write_file.write(cpp_code)
    write_file.close()

if __name__ == "__main__":
    main()