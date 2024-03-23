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
#include "print_progress.c"

// Macros
#define ARR_SIZE 2000000 // 2,000,000


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
int safeStringToInt(char str[], int *out);
int waitForThreads(pthread_t threads[], int numThreads);

// Main Loop, expected format ./<executable name> <number of threads>
int main(int argc, char* argv[]){   
    // Check that 2 Command Line arguments are provided, otherwise log not enough parameters, return -1
    if (argc != 2){
        printf("Failed to run the scheduling executable, must provide a command line argument of the number of threads to use\n");
        printf("Example: ./sched 4");
        return -1;
    }
    // Determine the number of threads to be used
    int numThreads;
    if (safeStringToInt(argv[1], &numThreads) != 0){
        printf("Failed to parse number of threads (%s) to int\n", argv[1]);
        return -1;
    }
    // Dynamically allocate arr of 2,000,000 ints
    int *junkInts = (int *)malloc(ARR_SIZE * sizeof(int));
    // create long long int totalSum to hold sum of arr with val 0
    long long int totalSum = 0;
    // create a mutex
    pthread_mutex_t mutex;
    if (pthread_mutex_init(&mutex, NULL) != 0){
        printf("Failed to initialize the mutex\n");
        return 1;
    }
    // Make an arr of len <cli Arg> of thread data
        // vals
            // localTid (thread ID) = should range from 0 - cliArg
            // pointer to prev allocated arr
            // numVals Corresponding to arr size
            // lock field pointing to created mutex
            // pointer to totalSum
    struct _thread_data_t thread_datas[numThreads];
    for (int i = 0; i < numThreads; i++){
        struct _thread_data_t newThreadData;
            newThreadData.data = &junkInts;
            newThreadData.localTid = i;
            newThreadData.lock = &mutex;
            newThreadData.numVals = ARR_SIZE;
            newThreadData.totalSum = &totalSum;
        thread_datas[i] = newThreadData;
    }
    // Make an arr of len <cli arg> of threads that each call arraySum with the corresponding thread data.
    pthread_t threads[numThreads];
    for (int i = 0; i < numThreads; i++){
        if (pthread_create(&threads[i], NULL, arraySum, (void*)&thread_datas[i]) != 0){
            printf("An error occured while trying to create a new thread\n");
            return 1;
        }
    }
    return 0;
    // join all the threads
    if (waitForThreads(threads, numThreads) != 0){
        printf("Failed to wait for threads\n");
        return 1;
    }
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
    print_progress(5,6);
}

int safeStringToInt(char str[], int *out){
    char *endptr;
    int result = strtol(str, &endptr, 10);
    if (*endptr != '\0') {
        printf("Failed to parse int\n");
        return 1;
    }
    *out = result;
    return 0;
}

int waitForThreads(pthread_t threads[], int numThreads){
    void* threadResult;
    for (int i =0; i < numThreads; i++){
        if ( pthread_join(threads[i], &threadResult) != 0){
            printf("An error occured in the main thread while waiting for a child thread.\n");
            return 1;
        }
        if (*(int*)threadResult != 0){
            printf("A thread exited with a non zero exit code.\n");
            return 1;
        }
    }
    return 0;
}
