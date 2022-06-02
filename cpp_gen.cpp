#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include "perf_events.hpp"
#include "common.h"

std::vector<PerfResults> results;

using namespace at;

int main() {
   CounterList counters;
   counters = get_defaults_counters();
   PerfEventsCounter couners_events(counters);
   couners_events.enable();
   PerfResults start_perf_results, end_perf_results, res;
   Tensor input;
   input = torch::ones({64, 1024} , TensorOptions(kCPU).dtype(torch::kFloat32)); 
   Tensor weight;
   weight = torch::ones({1024, 1024} , TensorOptions(kCPU).dtype(torch::kFloat32)); 
   Tensor bias;
   bias = torch::ones({1024} , TensorOptions(kCPU).dtype(torch::kFloat32)); 
   Tensor self1;
   self1 = torch::ones({64} , TensorOptions(kCPU).dtype(torch::kFloat32)); 

   couners_events.sample(start_perf_results);
   Tensor out0 = linear(input, weight, bias);
   couners_events.sample(end_perf_results);
   res = get_diff_counters(start_perf_results , end_perf_results);
   results.push_back(res);

   couners_events.sample(start_perf_results);
   Tensor out1 = mul(self1, 64);
   couners_events.sample(end_perf_results);
   res = get_diff_counters(start_perf_results , end_perf_results);
   results.push_back(res);

   dump_results_to_file(results);

   std::cout << "Done" << std::endl;
}