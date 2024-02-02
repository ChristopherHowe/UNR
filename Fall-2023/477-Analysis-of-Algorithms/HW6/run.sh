#!/bin/bash
set -e

numProblems=3

run_problem(){
    prob_num=$1
    if [ $prob_num -eq 1 ]; then
        g++ -o spendcost_partb spendcost_pb.cpp
        ./spendcost_partb
        rm spendcost_partb
    fi
    if [ $prob_num -eq 2 ]; then
        g++ -o spendcost_partc spendcost_pc.cpp
        ./spendcost_partc
        rm spendcost_partc
    fi
    if [ $prob_num -eq 3 ]; then
        g++ -o spendcost_partd spendcost_pd.cpp
        ./spendcost_partd
        rm spendcost_partd
    fi
}

if [ $1 -eq 0 ]; then
    for ((i = 1; i <= $numProblems; i++)); do
        run_problem $i
    done
else 
    run_problem $1
fi
