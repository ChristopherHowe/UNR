#!/bin/bash
set -e

info(){
    teal_color='\e[1;36m'
    white_color='\e[0m'
    echo -e "${teal_color}$@${white_color}"
}

test_file="10000000-random-numbers.txt"
num_threads=4
use_locking=0

run_loop=1
run_thread=1

run_many_count=1

script_dir=$PWD

while getopts ":lt:f:r:" opt; do
  case $opt in
    t)
      test_type=$OPTARG
      run_loop=0
      run_thread=0
      ;;
    f)
      run_many_count=$OPTARG
      ;;
    l)
      use_locking=1
      ;;
    r)
      num_threads=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

if [ -n "$test_type" ]; then
  case $test_type in
    1)
        run_loop=1
        ;;
    2)
        run_thread=1
        ;;
    *)
      echo "Invalid test type: $test_type. Supported values are 1 for looped_sum, 2 for threaded_sum, or no flag for both." >&2
      exit 1
      ;;
  esac
fi

if [ $run_loop -eq 1 ]; then
    info starting looped sum
    cd $script_dir/loop
    info making looped sum
    make
    total_time=0
    time_array=()

    for ((i=1; i<=$run_many_count; i++)); do
        info running looped sum, iteration: $i
        output=$(./looped_sum "../$test_file" | tee /dev/tty)
        time_val=$(echo "$output" | grep -oP 'Time taken \(ms\): \K[0-9]+\.[0-9]+')
        time_array+=("$time_val")
        total_time=$(echo "scale=3; $total_time + $time_val" | bc)
    done
    echo ${time_array[@]}
    echo Average time: $(echo "scale=3; $total_time / $run_many_count" | bc)  
    info finished looped sum
fi

if [ $run_thread -eq 1 ]; then
    info starting threaded sum
    cd $script_dir/thread
    info making threaded sum
    make
    
    time_array=()
    total_time=0

    for ((i=1; i<=$run_many_count; i++)); do
        info running threaded sum, iteration: $i
        output=$(./threaded_sum "$num_threads" "../$test_file" "$use_locking" | tee /dev/tty)
        time_val=$(echo "$output" | grep -oP 'Time taken \(ms\): \K[0-9]+\.[0-9]+')
        time_array+=("$time_val")
        total_time=$(echo "scale=3; $total_time + $time_val" | bc)
    done
    
    echo ${time_array[@]}
    echo Average time: $(echo "scale=3; $total_time / $run_many_count" | bc)  
    info finished threaded sum
fi
