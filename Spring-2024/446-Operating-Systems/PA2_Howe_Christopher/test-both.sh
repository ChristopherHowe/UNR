#!/bin/bash
set -e

test_file=oneThousand.txt
num_threads=4
use_locking=0

info(){
    teal_color='\e[1;36m'
    white_color='\e[0m'
    echo -e "${teal_color}$@${white_color}"
}


# info starting looped sum
cd loop
# info making looped sum
# make
# info running looped sum
# ./looped_sum "../$test_file"
# info finished looped sum

info starting threaded sum
cd ../thread
info making threaded sum
make
info running threaded sum
./threaded_sum "$num_threads" "../$test_file" "$use_locking"
info finished threaded sum
