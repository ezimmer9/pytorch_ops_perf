import os
import sys
import subprocess
from typing import final
import code_gen
import argparse
import shutil
import torch
from ops import get_ops_list


parser = argparse.ArgumentParser('PPET-main', description=__doc__)
parser.add_argument('--no-cmake', action='store_true', default=False,
                        help='disables CMake part')
parser.add_argument('--no-make', action='store_true', default=False,
                        help='disables make part')

def execute_cmd(cmd):
    lines = []
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
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

def main(args):
    try:
        print("\nConda enviroments is: {}\n".format(os.environ["CONDA_DEFAULT_ENV"]))
    except:
        print("\n----- you are not in conda env - make sure pytorch is install -----\n")
    
    ''' Code Generator part'''
    ops, consts = get_ops_list()
    code_gen.main(ops, consts)
    
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
    exe_cmd = ["./code_gen"]
    lines = execute_cmd(exe_cmd)
    strip_lines(lines)
    
    if lines[len(lines)-1] == "Done":
        print("\nThe execute success\n")
        print("The results are in build/perf_results\n\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
