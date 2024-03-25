#!/bin/bash
set -e

EXECUTABLE_NAME=sched
cd $PWD
make 

info(){
    teal_color='\e[1;36m'
    white_color='\e[0m'
    echo -e "${teal_color}$@${white_color}"
}

NUM_CORES=$(nproc --all)
echo Your CPU has $NUM_CORES Cores.

if [ -z "$1" ]; then
    echo please specify a problem demonstration to run.
    echo Problem 1: Run sched exec with same number of cores as CPU \($NUM_CORES\).
    echo Problem 2: Opens up three gnome terminal windows and runs the executable with some different scheduling methods.
    echo First window shows the running process latencies
    echo Second window shows the context switch watch output
    echo Third window shows which test is currently running.
    echo Tests ran: Realtime Round Robin Scheduling Real time FIFO Scheduling, Normal Scheduling, all with low, medium, and high priority.
    echo Problem 3: Run with one thread having its own CPU
    echo Problem 4: Run with combined steps of part 2 and 3
    exit
fi 

if [ "$1" -eq 1 ]; then
    info Executing test for problem 1 \(./sched $NUM_CORES\)
    ./sched 12
    exit
fi

if [ "$1" -eq 2 ]; then
    ./swap-scheduler-and-watch.sh
    ./sched 12
    exit
fi

if [ "$1" -eq 3 ]; then
    ./cpuset.sh
    ./sched 12
    exit
fi

if [ "$1" -eq 4 ]; then
    ./swap-scheduler-and-watch.sh
    ./cpuset.sh
    ./sched 12
    exit
fi
