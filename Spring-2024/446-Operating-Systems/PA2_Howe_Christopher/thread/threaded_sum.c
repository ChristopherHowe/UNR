/***************************************************
Author: Christopher Howe
Assignment Name: CS 446 Programming assignment 2
Date: 3/1/24
***************************************************/

// Preprocessor directives
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

#define MAX_NUM_INTS_TO_SUM 100000000

// Thread Data structure (Provided in assignment specifications)
typedef struct _thread_data_t {
    const int *data; //pointer to array of data read from file (ALL)
    int startInd; //starting index of thread’s slice
    int endInd; //ending index of thread’s slice
    long long int *totalSum; //pointer to the total sum variable in main
    pthread_mutex_t *lock; //critical region lock
} thread_data_t;

// Function Prototypes
int readFile(char filename[], int fileInts[]);
void fillThreadDataArray(
    thread_data_t threadDataObjs[],
    pthread_mutex_t* mutexPtr,
    int numThreads,
    int fileInts[],
    int numInts,
    long long int *totalSum
);
int getArrVal(int ind, thread_data_t *data);
void incrementTotalSum(long long int threadSum, thread_data_t *data);
void* arraySum(void* data);
int startThreads(pthread_t threads[], thread_data_t threadDataObjs[], int numThreads);
int waitForThreads(pthread_t threads[], int numThreads);
float getTVDiff(struct timeval tv1, struct timeval tv2);
void outputResult(long long int sum, float totalDurationMs, float runningDurationMs);
// Main Loop
int main(int argc, char* argv[]){    
    if (argc != 4){
        printf("Please enter a number of threads that you'd like to calculate with number of threads, filemane to open, and lock use value.\n");
        printf("(./threaded_sum 9 example.txt 0)\n");
        return -1;
    }
    int numThreads = atoi(argv[1]);
    int useLocks = atoi(argv[3]); // 0 for no 1 for yes.
    
    // NOTE: If this is statically allocated like `long long int fileInts[SIZE];` then seg faults may occur.
    // In order to circumvent this, allocating dynamically.
    int *fileInts = (int *)malloc(MAX_NUM_INTS_TO_SUM * sizeof(long long int));
    int numInts = readFile(argv[2], fileInts);
    if (numInts == -1){
        return 1;
    }
    // Setting a timer point to measure when the file reading finishes and thread overhead begins.
    struct timeval overHeadStart, threadStart, end;
    gettimeofday(&overHeadStart, NULL); 

    printf("Checking numthreads\n");
    if (numThreads > numInts){ // Check that the user didn't request more threads than the number of ints.
        printf("Too many threads requested\n");
        return -1;
    }

    // Set up the mutex
    pthread_mutex_t *mutex; // declare a pointer to the mutex
    if (useLocks){
        mutex = (pthread_mutex_t*) malloc (sizeof(pthread_mutex_t)); // request some space that the mutex pointer points to
        if (pthread_mutex_init(mutex, NULL) != 0){ // handle actually initializing the mutex
            printf("Failed to initialize the mutex\n");
            return 1;
        }
    } else {
        mutex = NULL;
    }

    // Initialize the total_sum that will be accessible by all the thread data objects.
    long long int totalSum = 0;

    // Create array of thread data.
    printf("Creating thread data\n");

    thread_data_t threadDataObjs[numThreads];
    fillThreadDataArray(threadDataObjs, mutex, numThreads, fileInts, numInts, &totalSum);
    // Start a timer to measure how long the threads actually take to execute
    gettimeofday(&threadStart, NULL);
    // Start all the threads
    
    printf("Starting threads\n");
    pthread_t threads[numThreads];
    if (startThreads(threads,threadDataObjs, numThreads) != 0){
        printf("Failed to start threads\n");
        return 1;
    }
    // Wait for all the threads to finish
    printf("Waiting for threads\n");
    if (waitForThreads(threads, numThreads) != 0){
        printf("Failed to wait for threads\n");
        return 1;
    }
    // set a timer endpoint.
    gettimeofday(&end, NULL);

    outputResult(totalSum, getTVDiff(overHeadStart, end), getTVDiff(threadStart,end));
    return 0;
}

int readFile(char filename[], int fileInts[]){
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("File not found...\n");
        return -1;
    }
    long count = 0;
    while (fscanf(file, "%d", &fileInts[count]) == 1) {
        count++;
        if (count > MAX_NUM_INTS_TO_SUM) {
            printf("Too many integers in the file. The Looped sum file only supports %d\n", MAX_NUM_INTS_TO_SUM);
            break;
        }
    }
    fclose(file);
    return count; 
}

void fillThreadDataArray(
    thread_data_t threadDataObjs[],
    pthread_mutex_t* mutexPtr,
    int numThreads,
    int fileInts[],
    int numInts,
    long long int *totalSum
){
    int remainderInts = numInts%numThreads;
    for (int i = 0; i < numThreads; i++){
        // printf("Creating thread data for thread %d\n", i);
        thread_data_t newThreadData;
            newThreadData.data = fileInts;
            newThreadData.startInd = i * numInts/numThreads;
            newThreadData.endInd = (i+1) * numInts/numThreads - 1;
            newThreadData.totalSum = totalSum;
            newThreadData.lock = mutexPtr;
        threadDataObjs[i] = newThreadData;
    }
    threadDataObjs[numThreads - 1].endInd += remainderInts;
}

int startThreads(pthread_t threads[], thread_data_t threadDataObjs[], int numThreads){
    for (int i = 0; i < numThreads; i++){
        // printf("Creating thread %d\n", i);
        if (pthread_create(&threads[i], NULL, arraySum, (void*)&threadDataObjs[i]) != 0){
            printf("An error occured while trying to create a new thread\n");
            return 1;
        }
    }
    return 0;
}

int waitForThreads(pthread_t threads[], int numThreads){
    void* threadResult;
    for (int i =0; i < numThreads; i++){
        // printf("Joining thread %d\n", i);
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

int getArrVal(int ind, thread_data_t *data){
    // NOTE: calling pthread_mutex_lock with null mutex is undefined behavior.
    pthread_mutex_t *l = data->lock;
    if (l == NULL){
        return data->data[ind];
    } else {
        // TODO: Check that instructions specify that reading requires locking.
        pthread_mutex_lock(l);
        int val = data->data[ind];
        pthread_mutex_unlock(l);
        return val;
    }
}

void incrementTotalSum(long long int threadSum, thread_data_t *data){
    // printf("Calling incrementTotalSum for threadedSum:%lld for thread starting at %d, with total being %lld before\n", threadSum, data->startInd, *(data->totalSum));
    pthread_mutex_t *l = data->lock;
    if (l == NULL){
        *(data->totalSum) += threadSum;
    } else {
        pthread_mutex_lock(l);
        *(data->totalSum) += threadSum;
        pthread_mutex_unlock(l);
    }
    // printf("Finished incrementTotalSum for threadedSum:%lld for thread starting at %d, with total now %lld\n", threadSum, data->startInd, *(data->totalSum));
}

void* arraySum(void* data){
    thread_data_t* threadData = (thread_data_t*) data;
    // printf("Calling array sum with start ind %d\n", threadData->startInd);
    long long int thread_sum = 0;
    for (int i = threadData->startInd; i <= threadData->endInd; i++){
        // if (i%50 == 0){
        //     printf("for thread starting at %d handling index %d\n", threadData->startInd, i);
        // }
        thread_sum += getArrVal(i,threadData);
    }
    incrementTotalSum(thread_sum, threadData);
    // printf("Finished array sum with start ind %d\n", threadData->startInd);
    int *res = (int*)malloc(sizeof(int));
    *res = 0;
    pthread_exit(res);  
}

// Takes the difference between two timeval structs
// assumes that tv2 is after tv1
// NOTE: Should be the same in both versions.
float getTVDiff(struct timeval tv1, struct timeval tv2){
    long secondPassed = tv2.tv_sec - tv1.tv_sec;
    long microsecondsPassed = tv2.tv_usec - tv1.tv_usec;
    return ((float)(secondPassed * 1000000 + microsecondsPassed)) / 1000;
}

void outputResult(long long int sum, float totalDurationMs, float runningDurationMs){
    printf("Total value of array: %lld\n", sum);
    printf("Time taken (ms): %.3f\n", totalDurationMs);
    printf("Time taken to run the threads (ms): %.3f\n", runningDurationMs);
}
