/***************************************************
Author: Christopher Howe
Assignment Name: CS 446 Programming assignment 2
Date: 3/1/24
***************************************************/

// Preprocessor directives
#include <sys/time.h>
#include <pthread.h>
#include "print_progress.c"

// Macros
#define ARR_SIZE 2000000 // 2,000,000

// Thread Data structure (Provided in assignment specifications)
typedef struct _thread_data_t {
    int localTid;
    const int *data;
    int numVals;
    pthread_mutex_t *lock;
    pthread_mutex_t *term_lock;
    long long int *totalSum;
} thread_data_t;

// Function Prototypes
// Required
void* arraySum(void* a);
// Additional
int safeStringToInt(char str[], int *out);
int startThreads(pthread_t threads[], thread_data_t threadDataObjs[], int numThreads);
int waitForThreads(pthread_t threads[], int numThreads);
int getTimeSpecDiff(struct timespec tv1, struct timespec tv2);
void fillThreadDataArray(thread_data_t thread_datas[], pthread_mutex_t* mutexPtr, int numThreads, int junkInts[], int numInts, long long int *totalSum);

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
    // Dynamically allocate arr of 2,000,000 ints,
    int *junkInts = (int *)malloc(ARR_SIZE * sizeof(int));
    for (int i = 0; i < ARR_SIZE; i++) {
        junkInts[i] = rand(); // Assign a random integer value
    }

    // create long long int totalSum to hold sum of arr with val 0
    long long int totalSum = 0;
        
    // create a mutex
    pthread_mutex_t mutex;
    if (pthread_mutex_init(&mutex, NULL) != 0){
        printf("Failed to initialize the mutex\n");
        return 1;
    }
    // Make an arr of len <cli Arg> of thread data
    struct _thread_data_t thread_datas[numThreads];
    fillThreadDataArray(thread_datas, &mutex, numThreads, junkInts, ARR_SIZE, &totalSum);
    
    // Make an arr of len <cli arg> of threads that each call arraySum with the corresponding thread data.
    pthread_t threads[numThreads];
    if (startThreads(threads, thread_datas, numThreads) != 0){
        printf("Failed to start all the threads\n");
        return 1;
    }
    // wait for all the threads to finish
    if (waitForThreads(threads, numThreads) != 0){
        printf("Failed to wait for threads\n");
        return 1;
    }

    free(junkInts);
    return 0;
}
// TODO: Remove average latency calculation, remove maxEver Stuff
void* arraySum(void* a){
    // This function should run indefinitely (inside while 1)
    thread_data_t* threadData = (thread_data_t*) a;
    while(1){
        // every loop of the while loop should sum all the values of the array and incrmement the totalSum value by the sum
        double maxLatency = 0; // Create a double storing the max latency
        long long int thread_sum = 0;
        for (int i = 0; i <= threadData->numVals; i++){
            struct timespec start, finish; // used to determine loop latency
            if (clock_gettime(CLOCK_REALTIME, &start) != 0){
                printf("Failed to get start timestamp");
            }
            // add a val from the arr to the local sum
            thread_sum += threadData->data[i];

            if (clock_gettime(CLOCK_REALTIME, &finish) != 0){
                printf("Failed to get finish timestamp");
            }
            // calculate the latency (duration) of the for loop iteration (ns) (long int)
            int latency = getTimeSpecDiff(start, finish);
            
            // update the max latency
            if (latency > maxLatency){
                maxLatency = latency;
            }
        }
        pthread_mutex_lock(threadData->lock);
        threadData->totalSum += thread_sum;
        // Print the max latency using the provided maxLatency object
        
        // NOTE: that the print_progress is called within the lock, this is due to the fact that stdout is a shared resource between the different
        // threads. If this is not protected in this way, undefined behavior occurs where occasionally threads may write to different lines than
        // they are supposed to. (See example screen shots)
        print_progress(threadData->localTid, maxLatency);
        pthread_mutex_unlock(threadData->lock);

    }
    pthread_exit(NULL);  
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

// Takes the difference between two timespec structs
// assumes that tv2 is after tv1
int getTimeSpecDiff(struct timespec tv1, struct timespec tv2){
    long secondPassed = tv2.tv_sec - tv1.tv_sec;
    long nanosecondsPassed = tv2.tv_nsec - tv1.tv_nsec;
    return (secondPassed * 1000000000 + nanosecondsPassed);
}

int startThreads(pthread_t threads[], thread_data_t threadDataObjs[], int numThreads){
    for (int i = 0; i < numThreads; i++){
        if (pthread_create(&threads[i], NULL, arraySum, (void*)&threadDataObjs[i]) != 0){
            printf("An error occured while trying to create a new thread\n");
            return 1;
        }
    }
    return 0;
}

void fillThreadDataArray(
    thread_data_t thread_datas[],
    pthread_mutex_t* mutexPtr,
    int numThreads,
    int junkInts[],
    int numInts,
    long long int *totalSum
){            
    for (int i = 0; i < numThreads; i++){
        struct _thread_data_t newThreadData;
            newThreadData.data = junkInts; // pointer to prev allocated arr
            newThreadData.localTid = i; // localTid (thread ID) = should range from 0 - cliArg
            newThreadData.lock = mutexPtr; // lock field pointing to created mutex
            newThreadData.numVals = ARR_SIZE; // numVals Corresponding to arr size
            newThreadData.totalSum = totalSum;
        thread_datas[i] = newThreadData;
    }
}
