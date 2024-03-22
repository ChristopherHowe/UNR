# Notes
## Available libraries
stdio, stdlib, string, pthread, sys/wait.h, sys/types.h, unistd.h, fcntl.h, errno.h, and sys/time.h

## Content
* normal task policies 
* real time task policies

### Completely Fair Scheduling (CFS)
Default scheduler in linux terminal since 2.6
Goal is to allocate CPU time evenly among processes
Dynamically adjusts priority
There is no control over static priority queues
Still allocates a higher priority for more interactive tasks and a lower priority for batch tasks
Takes into account a "niceness value"
* nicer processes (higher value) corresponds to higher priority (lower priority value)
* Processes that are more likely to yield the CPU have a higher niceness value.

