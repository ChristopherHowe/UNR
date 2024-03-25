# Project Notes
## Available libraries
stdio, stdlib, string, pthread, sys/wait.h, sys/types.h, unistd.h, fcntl.h, errno.h, and sys/time.h

## Requirements
No global variables
function prototypes are required
use locking (Should not be optional via command line arguments)
use method 2 of adjusting scheduling settings  (See content_notes.md)


## Goal
Develop a program that creates a user-controlled number of threads that will each be doing busy-wor and examine the comparative behavior of manipulating on of the threads scheduling properties and affinity.
observe the maximum latency and its variation indirectly, through a timing that happens within the fast inner for loop and its maximum value over each while loop iteration
