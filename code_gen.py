import os


def add_start_sample():
    return "   couners_events.sample(start_perf_results);\n"

def add_end_sample():
    s = "   couners_events.sample(end_perf_results);\n" + \
        "   res = get_diff_counters(start_perf_results , end_perf_results);\n" + \
        "   results.push_back(res);\n\n"
    return s


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
        cpp_code += "   {} {};\n".format(inp["type"], inp["name"])
        if isinstance(inp['shape'], list):
            str_list = [str(int_) for int_ in inp["shape"]]
            shape = "{" + ", ".join(str_list) + "}"
            if inp["type"] == "Tensor":
                cpp_code += "   {} = torch::ones({} , TensorOptions(kCPU).dtype(torch::{})); \n".format(inp["name"], shape, inp["dtype"])
            elif inp["type"] == "IntArrayRef":
                cpp_code += "   {} = {};\n".format(inp["name"], shape)
        else:
                cpp_code += "   {} = {};\n".format(inp["name"], inp["shape"])
            
    
    cpp_code += "\n"

    for op in ops:
        op_func = op[0]
        op_str = op_func + "(" + ", ".join(op[1:]) + ")"
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