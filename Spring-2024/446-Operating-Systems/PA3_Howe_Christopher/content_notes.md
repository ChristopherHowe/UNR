# Content Notes
## Completely Fair Scheduling (CFS)
Default scheduler in linux terminal since 2.6
Goal is to allocate CPU time evenly among processes
Dynamically adjusts priority
There is no control over static priority queues
Still allocates a higher priority for more interactive tasks and a lower priority for batch tasks
Takes into account a "niceness value"
* nicer processes (higher value) corresponds to higher priority (lower priority value)
* Processes that are more likely to yield the CPU have a higher niceness value.
* There is a bash command `nice` that allows you to incrememnt/decrement the niceness of a process

## normal task policies 
All normal task scheduling policies are managed by CFS
SCHED_OTHER= normal task scheduling policy that uses round robin with priority and niceness scheduling
Other normal scheduling processes  = SCHED_BATCH, SCHED_IDLE, SHED_DEADLINE
* used to indicate how priority weighting should be handled
## real time task policies
always able to premempt normal tasks, always have higher prio

use fix priority level policies
* priority range is from 0-99
    * for some reason 99 is the highest prio task   
* task with prio of 5 is placed in prio_5 queue
* if task is preempted ends back in the same prio queue
* tasks with a higher prio immediately preempt lower prio tasks

**SCHED_FIFO** = first method of implementing real time task prios, no preemption
**SCHED_RR** = second method of i8mpelemtning real time task prios, allows preemption at end of time quanta

## `chrt` Comand
used to manipulate scheduling policy and priority of a task.
- `chart -m` displays min/max prio for each policiy
- `sudo chrt -p -r <prio> <pid>` make task PID be a real time task with a certain prio with RR impementation
- `sudo chrt -p -f <prio> <pid>` make task PID be a real time task with a certain prio with FIFO impementation
- `sudo chrt -p -o 0 <pid>` switch task pid to Normal OTHER Policy

## Isolating CPUs
By default the Allocation of tasks and their assignment is handled by the CPU. There are two methods to do so.

We need to do so for this project so that our process can run almost exclusively on it's own CPU.
* move all processes (user and kernel spaces to a single CPU space with all but one CPU)
* Give single process its own CPU

### Using `isolcpus` and `taskset`
#### `isolcpus` in grub
can be used to isolate certain CPUs to control reservation of specific CPUs
accessed by editing the terminal
* adding isolcpus=0 isolates cpu 0, isolcpus=0-2 isolates cpus 0,1,2
* in order for this to be applied, `nano /etc/default/grub` and `sudo update-grub`
    * changes will be applied after booting.
#### `taskset` command
* `sudo taskset -p 0x00000001 <pid>` make the task associate with the PID use CPU 0
### Using `cpuset`
used to allocate certain tasks to certain CPU sets.
**CPU Set** = set of processes to run on a particular CPU.
Similar to `isolcpus` and `taskset` but has some limitiations, certain kernel tasks have to still have access to all CPUs.
How to install cpuset: `sudo apt install cpuset`

#### `cpuset` Commands
- `sudo cset set -l` lists all active cpu sets
- `sudo cset set -c <start CPU ind>-<end CPU ind> <new cpu set name>` create a new CPU set with name new name
    * instead of specifying a CPU range, can specify a single CPU.
- `sudo cset proc -m -f <original cpu set> -t <final cpu set>` moves all processes from the original cpu set to the final cpu set.
    * note: since cpuset command only operates on user level processes, if the original is root and the final is a freshly created set, then all user level processes will be moved from the root cpu set to the freshly created set.
    * can add the -k flag to move all kernel level tasks from the original to the final cpu set.
- `sudo cset proc -m -p <pid> -t <final cpu set>` = move a single process to a CPU set
- `sudo cset set -d <cpu set>` =  used to remove a CPU set. Processes still in the CPU set are assigned back to the root set.

## Additional notes
You can further fine tune [settings](https://documentation.suse.com/sles/12-SP5/html/SLES-all/cha-tuning-taskscheduler.html)
