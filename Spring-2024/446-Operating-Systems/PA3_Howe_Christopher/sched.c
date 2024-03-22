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

// ADD DEFINE STATEMENTS HERE

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
// Main Loop
int main(int argc, char* argv[]){   

}
