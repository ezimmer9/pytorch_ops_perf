import os
import sys
import subprocess
from typing import final
import code_gen
import argparse
import shutil
import torch
from ops import get_ops_list
from generate_code_from_json import generate_ops_from_json
import json

parser = argparse.ArgumentParser('PPET-main', description=__doc__)
parser.add_argument('--no-cmake', action='store_true', default=False,
                        help='disables CMake part')
parser.add_argument('--no-make', action='store_true', default=False,
                        help='disables make part')
parser.add_argument('-ppets', '--ppet_sweep', type=str, required=False, default=None,
                    dest='ppet_output_file', help="Run sweep with input file from ppet, which dump all the OPs not yet in compute ML model")
parser.add_argument('-sngl', '--singles', type=str, required=False,
                    dest="singles_json", help="In case of reading from JSON, Try to execute single OP at the time")

def execute_cmd(cmd, num_threads=0):
    lines = []
    if num_threads==0:
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    else:
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, env={"OMP_NUM_THREADS" : str(num_threads)})
    
    while True:
        nextline = p1.stdout.readline()
        if nextline == '' and p1.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
        lines.append(nextline)

    output = p1.communicate()[0]
    exitCode = p1.returncode

    if (exitCode == 0):
        p1.terminate()
        return lines
    else:
        raise Exception(cmd, exitCode, output)

def strip_lines(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip("\n")

def pt_cpp_main(args, ops_exter=None, consts_exter=None, num_threads=0):
    try:
        print("\nConda enviroments is: {}\n".format(os.environ["CONDA_DEFAULT_ENV"]))
    except:
        print("\n----- you are not in conda env - make sure pytorch is install -----\n")
    
    ''' Code Generator part'''
    if ops_exter is not None and consts_exter is not None:
        ops = ops_exter
        consts = consts_exter
    else:
        if args.ppet_output_file is not None:
            ops, consts = generate_ops_from_json(args)
        else:
            ops, consts = get_ops_list()
    code_gen.main(ops, consts)
    num_max_threads=torch.get_num_threads()
    ''' CMake part '''
    if not args.no_cmake:
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        pytorch_cmake = str(torch.utils.cmake_prefix_path)
        os.environ["CMAKE_PREFIX_PATH"] = pytorch_cmake
        #cmake_cmd = ["cmake", "-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`" , ".."]
        cmake_cmd = ["cmake", ".."]
        execute_cmd(cmake_cmd)
    
    ''' Make part'''
    if not args.no_make:
        if args.no_cmake:
            os.chdir("build")
        make_cmd = ["make"]
        execute_cmd(make_cmd)

    ''' Execute part'''
    if args.no_make and args.no_cmake:
        os.chdir("build")
    if num_threads==0:
        num_threads=1
    exe_cmd = ["./code_gen"]   # AG add threads    
    #exe_cmd = ["OMP_NUM_THREADS=",str(num_threads)," ./code_gen"]   # AG add threads
    #exe_cmd = ["./code_gen"]   # AG add threads
    lines = execute_cmd(exe_cmd, num_threads)
    strip_lines(lines)
    os.chdir("../")
    if lines[len(lines)-1] == "Done":
        print("\nThe execute success\n")
        print("The results are in build/perf_results\n\n")


if __name__ == "__main__":
    args = parser.parse_args()
    pt_cpp_main(args, ops_exter=None, consts_exter=None, num_threads=1)
