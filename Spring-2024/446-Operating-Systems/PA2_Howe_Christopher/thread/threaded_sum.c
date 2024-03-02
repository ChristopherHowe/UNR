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
        printf("Please provide the number of threads to be used and the filename to be summed as a command line argument.\n");
        printf("Example: ./looped_sum 4 data.txt.\n");
        return -1;
    }
    int numThreads = atoi(argv[1]);
    int useLocks = atoi(argv[3]); // 0 for no 1 for yes.
    
    // NOTE: If this is statically allocated like `long long int fileInts[SIZE];` then seg faults may occur.
    // In order to circumvent this, allocating dynamically.
    long long int *fileInts = (long long int *)malloc(MAX_NUM_INTS_TO_SUM * sizeof(long long int));
    int numInts = readFile(argv[2], fileInts);
    if (numInts == -1){
        return 1;
    }
    // Starting timer now since the overhead of setting up threads is part of the difference between it and a non threaded approach.
    // Looped approach starts timer after reading file input too.
    long long int microsecStart = getTime();    

    if (numThreads > numInts){ // Check that the user didn't request more threads than the number of ints.
        printf("Too many threads requested\n");
        return -1;
    }

    // Set up the mutex
    pthread_mutex_t *mutex;
    if (useLocks){
        if (pthread_mutex_init(mutex, NULL) == 0){
            printf("Failed to initialize the mutex\n");
            exit(1);
        }
    } else {
        mutex = NULL;
    }

    // initialize the total_sum.
    long long int totalSum = 0;

    // create the array of thread objects.
    pthread_t threads[numThreads];
    for (int i = 0; i < numThreads; i++){
        thread_data_t newThreadData;
            newThreadData.data = fileInts;
            newThreadData.startInd = i * 25;
            newThreadData.endInd = (i+1) * 25 - 1;
            newThreadData.totalSum = &totalSum;
            newThreadData.lock = mutex;
        pthread_create(threads[i], NULL, arraySum, NULL);
    }

    // stop the timer
    long long int microsecFinish = getTime();

    
    return 0;
}

int readFile(char filename[], int fileInts[]){
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("File not found...\n");
        return -1;
    }
    long count = 0;
    while (fscanf(file, "%lld", &fileInts[count]) == 1) {
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

void* arraySum(void*){

}
