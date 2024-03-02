/***************************************************
Author: Christopher Howe
Assignment Name: CS 446 Programming assignment 2
Date: 3/1/24
***************************************************/

// Preprocessor directives

#include <stdio.h>
#include <stdlib.h> // used for atoi
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
int getTime();
void* arraySum(void*);

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
    // Starting timer now since the overhead of setting up threads is part of the difference between it and a non threaded approach.
    // Looped approach starts timer after reading file input too.
    // TODO: Make sure matches instructions
    long long int microsecStart = getTime();    

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
            exit(1);
        }
        printf("Created the mutex\n");
    } else {
        mutex = NULL;
    }

    // initialize the total_sum.
    long long int totalSum = 0;

    // Create array of thread data.
    struct _thread_data_t threadDataObjs[numThreads];
    for (int i = 0; i < numThreads; i++){
        printf("Creating thread data for thread %d\n", i);
        thread_data_t newThreadData;
            newThreadData.data = fileInts;
            newThreadData.startInd = i * 25;
            newThreadData.endInd = (i+1) * 25 - 1;
            newThreadData.totalSum = &totalSum;
            newThreadData.lock = mutex;
        threadDataObjs[i] = newThreadData;
    }
    
    // create the array of thread objects.
    pthread_t threads[numThreads];
    for (int i = 0; i < numThreads; i++){
        printf("Creating thread %d\n", i);
        if (pthread_create(&threads[i], NULL, arraySum, (void*)&threadDataObjs[i]) != 0){
            printf("An error occured while trying to create a new thread\n");
            return 1;
        }
    }

    void* threadResult;
    for (int i =0; i < numThreads; i++){
        printf("Joining thread %d\n", i);
        if ( pthread_join(threads[i], &threadResult) != 0){
            printf("An error occured in the main thread while waiting for a child thread.\n");
            return 1;
        }
    }

    // stop the timer
    long long int microsecFinish = getTime();
    printf("Total value of array: %lld\n",totalSum);
    float durationMillis = ((float)(microsecFinish - microsecStart)) / 1000; // NOTE: only have 3 digits of precision since microseconds 
    printf("Time taken (ms): %.3f\n", durationMillis);
    
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


int getTime(){
    struct timeval tv;
    gettimeofday(&tv, NULL); 
    return tv.tv_usec;
}

int getArrVal(int ind, struct _thread_data_t *data){
    pthread_mutex_t *l = data->lock;
    // calling pthread_mutex_lock with null mutex is undefined behavior.
    if (l == NULL){
        return data->data[ind];
    } else {
        pthread_mutex_lock(l);
        int val = data->data[ind];
        pthread_mutex_unlock(l);
        return val;
    }
}

void* arraySum(void* data){
    struct _thread_data_t* threadData = (struct _thread_data_t*) data;
    printf("Calling array sum with start ind %d\n", threadData->startInd);
    long long int thread_sum = 0;
    for (int i = threadData->startInd; i <= threadData->endInd; i++){
        printf("threaded sum was %lld before adding ind %d\n", thread_sum, i);
        thread_sum += getArrVal(i,threadData);
    }
    printf("Finished array sum with start ind %d\n", threadData->startInd);
    return NULL;
}
