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

exit
fi 

if [ "$1" -eq 1 ]; then
    info Executing test for problem 1 \(./sched $NUM_CORES\)
    ./$EXECUTABLE_NAME 11
    exit
fi

realtime_RR(){
    local prio=$1
    local pid=$2
    sudo chrt -p -r $prio $pid
}

realtime_FIFO(){
    local prio=$1
    local pid=$2
    sudo chrt -p -f $prio $pid
}

normal_scheduling(){
    pid=$1
    sudo chrt -p -o 0 $pid
}

LOW_PRIO=0
MED_PRIO=50
HIGH_PRIO=99

iterate_tests(){
    sleep 1 # wait for executable to start
    echo checking for name $EXECUTABLE_NAME
    PROCESS_PID=$(pgrep $EXECUTABLE_NAME)

    gnome-terminal -- watch -n .5 grep ctxt /proc/$PROCESS_PID/status

    echo "Running $EXECUTABLE_NAME with 10 cores, PID=$PROCESS_PID"

    read -p "Press enter to start real time with round robin and low priority"
    realtime_RR $LOW_PRIO $PROCESS_PID

    read -p "Press enter to start real time with round robin and medium priority"
    realtime_RR $MED_PRIO $PROCESS_PID

    read -p "Press enter to start real time with round robin and high priority"
    realtime_RR $HIGH_PRIO $PROCESS_PID

    read -p "Press enter to start real time with FIFO and low priority"
    realtime_FIFO $LOW_PRIO $PROCESS_PID

    read -p "Press enter to start real time with FIFO and medium priority"
    realtime_FIFO $MED_PRIO $PROCESS_PID
    
    read -p "Press enter to start real time with FIFO and high priority"
    realtime_FIFO $HIGH_PRIO $PROCESS_PID

    read -p "Press enter to start normal scheduling"
    normal_scheduling $PROCESS_PID
    
    read -p "Press Enter to exit..."
}

if [ "$1" -eq 2 ]; then
    export -f iterate_tests
    export -f realtime_RR
    export -f realtime_FIFO
    export -f normal_scheduling
    export EXECUTABLE_NAME=$EXECUTABLE_NAME
    info "Executing test for problem 2."
    THIS=$BASH_SOURCE
    gnome-terminal --window -- bash -c 'iterate_tests; bash'
    # Start the process
    ./$EXECUTABLE_NAME 10
   
fi
