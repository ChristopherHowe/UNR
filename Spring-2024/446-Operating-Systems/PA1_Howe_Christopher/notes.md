# Requirements
* write a shell script
* Makefile with at least one target.

# Part 1
## Goal: Create a stripped down shell

* **Prompt** = username@machineName: currentDirectory
* When commands are executed from the prompt, the shell creates child processes.

### How the shell handles commands
1. print the prompt
2. get the input
3. parse the input
4. find files associated with the command (PATH)
5. pass arguments from the shell to the program
6. execute the command with the arguments


