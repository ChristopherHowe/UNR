#!/bin/bash
set -e

run_test(){
    sleep 1
    PID=$(pgrep sched)
    echo Running tests for sched with PID=$PID

    #Get the SPIDs (Thread IDs) associated with the process ID
    spids=($(ps -T -p $PID | awk 'NR > 1 {print $2}'))

    # Print the array elements (SPIDs)
    echo sched SPIDs \(TIDs\)
    echo "${spids[@]}"

    echo active cpu sets
    sudo cset set -l

    echo make system CPU set for all tasks
    sudo cset set -c 0-10 system

    echo create the dedicated CPU set
    sudo cset set -c 11 dedicated

    echo move all user level and kernel level tasks to the new system CPU set
    sudo cset proc -m -f root -t system
    sudo cset proc -k -f root -t system

    echo give a single thread \(${spids[1]}\) an entire CPU core
    sudo cset proc -m -p ${spids[1]} -t dedicated
    
    read -p "Press enter to reset"
    cset set -d dedicated
    cset set -d system

}

main(){
    export -f run_test
    gnome-terminal --window -- bash -c 'run_test; bash'
}

main $1
