#include <iostream>

#include "perf_events.hpp"
#include <linux/perf_event.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <math.h>
#include <inttypes.h>
#include <sys/types.h>
#include <asm/unistd.h>

//#define DBG_PRINT(x) x
#define DBG_PRINT(x)

long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    int ret;
    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    return ret;
}

PerfHWCounter::PerfHWCounter(
    const std::string &name, 
    PerfType perf_type, 
    __u64 perf_id, 
    PerfHWCounter *group_leader):_name(name)
{
    ////////////////////////////
    // HW instructions counter enable
    struct perf_event_attr pe;
    bool is_leader = group_leader == nullptr;

    DBG_PRINT(std::cout <<">>> opening " << name << ", type:" <<perf_type << ", id: " << perf_id << " as " << (is_leader ? "leader\n": "follower\n"));

    memset(&pe, 0, sizeof(struct perf_event_attr));
    pe.type = perf_type; // PERF_TYPE_HARDWARE;
    pe.size = sizeof(struct perf_event_attr);
    pe.config = perf_id;
    pe.disabled = is_leader ? 1 : 0;
    pe.exclude_kernel = 1;

    //pe.pinned = 1; // ranc: try to pinn in PMU
    //pe.exclusive = 0;
    // Don't count hypervisor events.
    pe.exclude_hv = 1;
    pe.inherit=0;

    int group_fd = is_leader ? -1 : group_leader->_fd;
    
    _fd = perf_event_open(&pe, 0, -1, group_fd, 0);
    if (_fd == -1) {
        perror("Error opening perf_event");
        fprintf(stderr, "Error opening perf_event %s\n", name.c_str());
        _is_active=false;
        return;
    }        
    _is_active=true;
}

PerfHWCounter::~PerfHWCounter()
{
    DBG_PRINT(std::cout <<">!!! closing " << _name << "\n");
    if (_is_active)
    {
        close(_fd);
        _is_active=false;
    }
}

void PerfHWCounter::enable()
{
    if (_is_active)
    {
        //std::cout <<">>> enabling " << _name << "\n";
        ioctl(_fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(_fd, PERF_EVENT_IOC_ENABLE, 0);
    }
}

void PerfHWCounter::disable()
{
    if (_is_active)
    {
        //std::cout <<">>> disabling " << _name << "\n";
        ioctl(_fd, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(_fd, PERF_EVENT_IOC_RESET, 0);
    }
}

double PerfHWCounter::sample()
{
    if (!_is_active) return 0;
    long long count;
    auto res = read(_fd, &count, sizeof(long long));
    if (res != sizeof(long long))
    {
        std::cerr << "Error reading counter " << _name << "\n";
        close(_fd);
        _is_active=false;
        return 0;
    }
    //std::cout <<">>> sampling " << _name << ", res: " << res << ", value: "<< count << "\n";
    return (float)count;
}


/***********************************************************/
/**** PerfEventsCounter  ***********************************/
/***********************************************************/

PerfEventsCounter::PerfEventsCounter(const CounterList &counters)    
{      
    //add_counter("instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);
    //add_counter("cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
    //add_counter("llc_access", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
    //add_counter("llc_miss", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);
    //add_counter("llc_read", PERF_TYPE_HW_CACHE, (0x2 | 0x0<<8 | 0x0<<16));
    _group_leader = nullptr;
    for (auto it : counters)    
    {
        add_counter(it.first, it.second.first, it.second.second);
    }
}

void PerfEventsCounter::add_counter(const std::string name, enum perf_type_id perf_type, __u64 perf_id)
{    
    auto it = _counters.emplace(std::piecewise_construct, std::make_tuple(name), std::make_tuple(name, perf_type, perf_id, _group_leader)); 
    if (_group_leader == nullptr)
    {
        _group_leader = &it.first->second;
    }
}


void PerfEventsCounter::enable()
{
    if (_group_leader != nullptr)
    {
        _group_leader->enable();
    }
    /*
    for (auto &it : _counters)
    {
        it.second.enable();            
    }
    */
}

void PerfEventsCounter::disable()
{ 
    if (_group_leader != nullptr)
    {
        _group_leader->disable();
    }

    /*
    for (auto &it : _counters)
    {
        it.second.disable();
    }
    */
}

void PerfEventsCounter::sample(std::map<std::string, double> &ret)
{
    for (auto &it : _counters)
    {
        float f = it.second.sample();
        //std::cout <<" #*#*#*#* sampling counter " << it.first << " = "<< f << "\n";
        ret[it.first]=f;
    }
}
