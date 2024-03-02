# Goal
implement a general threading API and compare threading preformance to looped preformance.
Write a non threaded program that sums an array and tracks how long
write a threaded implementation and track it
respond to the questions


## Aceptable Libraries
stdio
stdlib
string
pthread
ctype
sys
time.h

## Specifications
* Program should be able to accept up to 100,000,000 values


## Notes on how threading works
The first thread is the main thread
all other threads are made using pthread_create
Parent thread should call pthread_join to wait for all the child threads to finish.

