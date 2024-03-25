#!/bin/bash

pid=$1

get_total_ctxt_switches() {
    local pid=$1
    local voluntary=$(grep "^voluntary_ctxt_switches" /proc/"$pid"/status | awk '{print $2}')
    local nonvoluntary=$(grep "nonvoluntary_ctxt_switches" /proc/"$pid"/status | awk '{print $2}')
    echo $((voluntary + nonvoluntary))
}

samples=()
num_samples=0
lastTickVal=$(get_total_ctxt_switches $pid)
while true; do
    currentTotalCTXSwitches=($(get_total_ctxt_switches $pid))
    difference=$((currentTotalCTXSwitches- lastTickVal))
    lastTickVal=$currentTotalCTXSwitches
    samples+=($difference)
    if [ $num_samples -eq "15" ]; then 
        samples=("${samples[@]:1}")
    else
        num_samples=$((num_samples + 1))
    fi

    sum=0
    for val in "${samples[@]}"; do
        sum=$((sum + val))
    done
    clear
    tput cup 0 0
    echo Watching $pid.
    echo Average Number of context switches per second \(15 sec window\): $(echo "scale=2; $sum / $num_samples" | bc)
    echo Samples: ${samples[@]}
    grep ctxt /proc/"$pid"/status
    sleep 1
done
