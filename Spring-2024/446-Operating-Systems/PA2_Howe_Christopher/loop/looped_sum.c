/***************************************************
Author: Christopher Howe
Assignment Name: CS 446 Programming assignment 2
Date: 3/1/24
***************************************************/

// Preprocessor directives
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_NUM_INTS_TO_SUM 100000000 // From Specification

// Function Prototypes
// Required
int readFile(char filename[], int fileInts[]);
long long int sumArray(long long int fileInts[], int numInts);
// Additional
float getTVDiff(struct timeval tv1, struct timeval tv2);
void IntArrToLongLongArr(int in[], long long int out[], int numVals);
void outputResult(long long int sum, float totalDurationMs);

// Main Loop
int main(int argc, char* argv[]){    
    if (argc != 2){
        printf("Please provide the filename to be summed as a command line argument.\n");
        printf("Example: ./looped_sum data.txt.\n");
        return 1;
    }
    // NOTE: If this is statically allocated like `long long int fileInts[SIZE];` then seg faults may occur.
    // In order to circumvent this, allocating dynamically.
    int *fileInts = (int *)malloc(MAX_NUM_INTS_TO_SUM * sizeof(int));
    int numInts = readFile(argv[1], fileInts);
    if (numInts == -1){
        return 1;
    }
    
    long long int *inputVals = (long long int *)malloc(MAX_NUM_INTS_TO_SUM * sizeof(long long int));
    IntArrToLongLongArr(fileInts, inputVals, numInts);
    free(fileInts); // No longer require the fileInts array.


    struct timeval start, end;
    gettimeofday(&start, NULL); 
    long long int sum = sumArray(inputVals, numInts);
    gettimeofday(&end, NULL);    

    free(inputVals);

    printf("Total value of array: %lld\n",sum);
    printf("Time taken (ms): %.3f\n", getTVDiff(start, end));
    return 0;
}

// Reads a file of integers to be used as input
// NOTE: Should be the same in both versions.
int readFile(char filename[], int fileInts[]){
    printf("Reading data...\n");
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("File not found...\n");
        return -1;
    }
    long count = 0;
    while (fscanf(file, "%d", &fileInts[count]) == 1) {
        count++;
        if (count > MAX_NUM_INTS_TO_SUM) {
            printf("Too many integers in the file. This program only supports up to %d values\n", MAX_NUM_INTS_TO_SUM);
            break;
        }
    }
    fclose(file);
    printf("Finished Reading Data\n");
    return count; 
}

// Takes the difference between two timeval structs
// assumes that tv2 is after tv1
// NOTE: Should be the same in the both versions.
float getTVDiff(struct timeval tv1, struct timeval tv2){
    long secondPassed = tv2.tv_sec - tv1.tv_sec;
    long microsecondsPassed = tv2.tv_usec - tv1.tv_usec;
    return ((float)(secondPassed * 1000000 + microsecondsPassed)) / 1000;
}

long long int sumArray(long long int fileInts[], int numInts){
    long long int sum = 0;
    for (int i=0; i < numInts; i++){
        sum += fileInts[i];
    }
    return sum;
}

// NOTE: It appears that since the readFile Function expects a int[] and sumArray expects a long long int[], the
// array must be converted. The instructions dictate that the defined parameter types must be used.
void IntArrToLongLongArr(int in[], long long int out[], int numVals){
    for(int i=0; i<numVals; i++){
        out[i] = (long long int)in[i];
    }
}

void outputResult(long long int sum, float totalDurationMs){
    printf("Total value of array: %lld\n", sum);
    printf("Time taken (ms): %.3f\n", totalDurationMs);
}
