#!/bin/bash
iterate_tests(){
    realtime_RR(){
        local prio=$1
        local pid=$2
        sudo chrt -p -r $prio $pid
        chrt -p $pid

    }

    realtime_FIFO(){
        local prio=$1
        local pid=$2
        sudo chrt -p -f $prio $pid
        chrt -p $pid

    }

    normal_scheduling(){
        pid=$1
        sudo chrt -p -o 0 $pid
        chrt -p $pid

    }

    sleep 1
    PID=$(pgrep sched)
    echo Running tests for sched with PID=$PID


    #Get the SPIDs (Thread IDs) associated with the process ID 172216
    spids=($(ps -T -p $PID | awk 'NR > 1 {print $2}'))

    # Print the array elements (SPIDs)
    echo sched SPIDs \(TIDs\)
    echo "${spids[@]}"
    
    TID=${spids[1]}
    echo Watching Thread 0 \($TID\) and adjusting prios 

    sleep 1 # wait for executable to start
    
    gnome-terminal -- ./get-average-ctx-switches.sh $TID

    LOW_PRIO="1"
    MED_PRIO="50"
    HIGH_PRIO="99"

    read -p "Press enter to start real time with round robin and low priority"
    realtime_RR $LOW_PRIO $TID

    read -p "Press enter to start real time with round robin and medium priority"
    realtime_RR $MED_PRIO $TID

    read -p "Press enter to start real time with round robin and high priority"
    realtime_RR $HIGH_PRIO $TID

    read -p "Press enter to start real time with FIFO and low priority"
    realtime_FIFO $LOW_PRIO $TID

    read -p "Press enter to start real time with FIFO and medium priority"
    realtime_FIFO $MED_PRIO $TID
    
    read -p "Press enter to start real time with FIFO and high priority"
    realtime_FIFO $HIGH_PRIO $TID

    read -p "Press enter to start normal scheduling"
    normal_scheduling $TID
    
    read -p "Press Enter to exit..."
}

main(){
    export -f iterate_tests
    info "Executing test for problem 2."
    THIS=$BASH_SOURCE
    gnome-terminal --window -- bash -c 'iterate_tests; bash'
}

main $1
