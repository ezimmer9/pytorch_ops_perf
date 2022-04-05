#pragma once
#ifndef __PPTE_EVENTS_HPP
#define __PPTE_EVENTS_HPP


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

#include <map>

typedef enum perf_type_id PerfType;
typedef std::map<std::string, std::pair<enum perf_type_id, __u64>> CounterList;

class PerfHWCounter {
    bool _is_active;
    long _fd;
    std::string _name;

public:
    PerfHWCounter() = delete;
    PerfHWCounter(const PerfHWCounter &) = delete;
    PerfHWCounter(const std::string &name, PerfType perf_type, __u64 perf_id, PerfHWCounter *group_leader=nullptr);
    ~PerfHWCounter();

    void enable();
    void disable();

    double sample();

    inline bool is_active() const { return _is_active;}
};


class PerfEventsCounter
{
private: 
    std::map<std::string, PerfHWCounter> _counters;
    PerfHWCounter *_group_leader;
    
public:
    PerfEventsCounter(const CounterList &counters);
    void add_counter(const std::string name, enum perf_type_id perf_type, __u64 perf_id);
        
    void  sample(std::map<std::string, double> &ret);
    void  enable();
    void  disable();
};

#endif
