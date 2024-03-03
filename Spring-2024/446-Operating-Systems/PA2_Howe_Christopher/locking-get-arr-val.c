#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>

/*
PURPOSE: This file exists as a backup in case I want to use locking for the entire array.
*/

// Thread Data structure (Provided in assignment specifications)
typedef struct _thread_data_t {
    const int *data; //pointer to array of data read from file (ALL)
    int startInd; //starting index of thread’s slice
    int endInd; //ending index of thread’s slice
    long long int *totalSum; //pointer to the total sum variable in main
    pthread_mutex_t *lock; //critical region lock
} thread_data_t;

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
