/***************************************************
Author: Christopher Howe
Assignment Name: CS 446 Programming assignment 2
Date: 3/1/24
***************************************************/

// Preprocessor directives
// #include <stdio.h>
// #include <stdlib.h>
// #include <sys/time.h>
#include <pthread.h>

// Macros

// Thread Data structure (Provided in assignment specifications)
typedef struct _thread_data_t {
    int localTid;
    const int *data;
    int numVals;pthread_mutex_t *lock;
    long long int *totalSum;
} thread_data_t;

// Function Prototypes
// Required
void* arraySum(void* a);
// Additional


// Main Loop
int main(int argc, char* argv[]){   
// Check that 2 Command Line arguments are provided, otherwise log not enough parameters, return -1
// Dynamically allocate arr of 2,000,000 ints
// create long long int totalSum to hold sum of arr with val 0
// create a mutex
// Make an arr of len <cli Arg> of thread data
    // vals
        // localTid (thread ID) = should range from 0 - cliArg
        // pointer to prev allocated arr
        // numVals Corresponding to arr size
        // lock field pointing to created mutex
        // pointer to totalSum
// Make an arr of len <cli arg> of threads that each call arraySum with the corresponding thread data.
// join all the threads
}

void* arraySum(void* a){
// This function should run indefinitely (inside while 1)
    // every loop of the while loop should sum all the values of the array and incrmement the totalSum value by the sum
    // Create a double storing the max latency
    // For each iteration of the for loop
        // add a val from the arr to the localSum
        // calculate the latency of the for loop
            // create a struct timespec object at the beggining of the loop using clock_gettime
            // create a struct timespec object at the end of the loop using clock_gettime
            // calculate the duration of the for loop iteration (ns) (long int)
            // update the max latency
    // For each iteration of the while loop extract the max latency for all the for loop iterations
    // print the max latency with print_progress(pid_t local_tid, size_t value)
}
