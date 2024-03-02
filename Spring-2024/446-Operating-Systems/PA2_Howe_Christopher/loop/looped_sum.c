/***************************************************
Author: Christopher Howe
Assignment Name: CS 446 Programming assignment 2
Date: 3/1/24
***************************************************/

// Preprocessor directives

#include <stdio.h>
#include <stdlib.h> // used for malloc
#include <sys/time.h>

#define MAX_NUM_INTS_TO_SUM 100000000

// Function Prototypes
int readFile(char filename[], long long int fileInts[]);
int getTime();
long long int sumArray(long long int  fileInts[], int numInts);

// Main Loop
int main(int argc, char* argv[]){    
    printf("Starting Main\n");
    if (argc != 2){
        printf("Please provide the filename to be summed as a command line argument.\n");
        printf("Example: ./looped_sum data.txt.\n");
        return 1;
    }
    // NOTE: If this is statically allocated like `long long int fileInts[SIZE];` then seg faults may occur.
    // In order to circumvent this, allocating dynamically.
    long long int *fileInts = (long long int *)malloc(MAX_NUM_INTS_TO_SUM * sizeof(long long int));
    
    int numInts = readFile(argv[1], fileInts);
    if (numInts == -1){
        return 1;
    }
    long long int microsecStart = getTime();    
    long long int sum = sumArray(fileInts, numInts);
    long long int microsecFinish = getTime();
    
    free(fileInts);// Don't forget to free the int array when its done.
    
    printf("Total value of array: %d\n",sum);
    float durationMillis = ((float)(microsecFinish - microsecStart)) / 1000; // NOTE: only have 3 digits of precision since microseconds 
    printf("Time taken (ms): %.3f\n", durationMillis);
    return 0;
}


int readFile(char filename[], long long int fileInts[]){
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

int getMaxIntSize(){
    if (sizeof(int) == 2) {
        return 32767;
    }
    else if (sizeof(int) == 4) {
        return 2147483647;
    }
    else {
        printf("The sizeof type int was unexpected\n");
        return -1;
    }
}
long long int sumArray(long long int fileInts[], int numInts){
    long long int sum = 0;
    for (int i=0; i < numInts; i++){
        sum += fileInts[i];
    }
    return sum;
}

