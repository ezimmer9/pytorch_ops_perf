#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "perf_events.hpp"

typedef std::map<std::string, double> PerfResults;
#define NUM_COUNTERS 5

/*
PERF_TYPE_HARDWARE = 0
PERF_TYPE_SOFTWARE 	
PERF_TYPE_TRACEPOINT 	
PERF_TYPE_HW_CACHE 	
PERF_TYPE_RAW 	
PERF_TYPE_BREAKPOINT 	
PERF_TYPE_MAX 

PERF_COUNT_HW_CPU_CYCLES = 0	
PERF_COUNT_HW_INSTRUCTIONS 	
PERF_COUNT_HW_CACHE_REFERENCES 	
PERF_COUNT_HW_CACHE_MISSES 	
PERF_COUNT_HW_BRANCH_INSTRUCTIONS 	
PERF_COUNT_HW_BRANCH_MISSES 	
PERF_COUNT_HW_BUS_CYCLES 	
PERF_COUNT_HW_STALLED_CYCLES_FRONTEND 	
PERF_COUNT_HW_STALLED_CYCLES_BACKEND 	
PERF_COUNT_HW_REF_CPU_CYCLES 	
PERF_COUNT_HW_MAX 

PERF_COUNT_HW_CACHE_L1D = 0	
PERF_COUNT_HW_CACHE_L1I 	
PERF_COUNT_HW_CACHE_LL 	
PERF_COUNT_HW_CACHE_DTLB 	
PERF_COUNT_HW_CACHE_ITLB 	
PERF_COUNT_HW_CACHE_BPU 	
PERF_COUNT_HW_CACHE_NODE 	
PERF_COUNT_HW_CACHE_MAX 

ERF_COUNT_SW_CPU_CLOCK = 0
PERF_COUNT_SW_TASK_CLOCK 	
PERF_COUNT_SW_PAGE_FAULTS 	
PERF_COUNT_SW_CONTEXT_SWITCHES 	
PERF_COUNT_SW_CPU_MIGRATIONS 	
PERF_COUNT_SW_PAGE_FAULTS_MIN 	
PERF_COUNT_SW_PAGE_FAULTS_MAJ 	
PERF_COUNT_SW_ALIGNMENT_FAULTS 	
PERF_COUNT_SW_EMULATION_FAULTS 	
PERF_COUNT_SW_MAX
*/


CounterList get_defaults_counters()
{
    CounterList counters;
    std::string str[NUM_COUNTERS] = {"instructions" , "cycles" , "llc_access" , "L1_read" , "L1_write"};
    enum perf_type_id types[NUM_COUNTERS] = {PERF_TYPE_HARDWARE , PERF_TYPE_HARDWARE , PERF_TYPE_HARDWARE , PERF_TYPE_HW_CACHE , PERF_TYPE_HW_CACHE};
    __u64 id[NUM_COUNTERS] = {PERF_COUNT_HW_INSTRUCTIONS , PERF_COUNT_HW_CPU_CYCLES , PERF_COUNT_HW_CACHE_REFERENCES , 0x0 , 0x2};
    for (int i = 0 ; i < NUM_COUNTERS ; i++){
        counters.insert(std::make_pair(str[i], std::make_pair(types[i], id[i])));
    }
    return counters;
}

PerfResults get_diff_counters(PerfResults start , PerfResults end)
{
    PerfResults ret;
    for (auto &it: start)
    {
        ret[it.first] = end[it.first] - start[it.first];
    }
    return ret;
}

void dump_results_to_file(std::vector<PerfResults> res)
{
    uint index = 0;
    std::ofstream myfile;
    myfile.open ("perf_results.txt");
    myfile << "{" << std::endl;
    uint res_size = res.size();
    for (auto &vec: res){
        myfile << '"' << index << '"' << ": " << "{";
        uint size = vec.size();
        uint counter = 0;
        for (auto &it: vec)
        {
            myfile << '"' << it.first << '"' << ":" <<  it.second;
            if (counter < size-1) {
                myfile << ", ";
            }
            counter++;
        }
        myfile << "}";
        if (index < res_size-1){
            myfile << ", ";
        }
        myfile << std::endl;
        index += 1;
    }
    myfile << "}" << std::endl;
    myfile.close();
}
