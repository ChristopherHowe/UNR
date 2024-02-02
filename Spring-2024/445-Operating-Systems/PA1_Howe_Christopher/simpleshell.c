/*
Author: Chris Howe
Date: Febuary 1 2024
Title: CS 446 PA1
*/

// Preprocessor Directives
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/wait.h>



// The following libraries can be used
//stdio.h, string.h, stdlib.h, sys/wait.h, sys/types.h, unistd.h, fcntl.h, errno.h, sys/stat.h

// display functions: write() and printf()
// reading functions: fgets() scanf()
// sys call: write
// clib: printf, fgets, scanf

// #define OUT_FILE "spendcost_pb_out.txt"

// commands that need to be supported:
// exit, cd, all commands supported by execvp
// execvp supports ls -l and clear

#define MAX_TOKENS 20
#define MAX_TOKEN_LEN 20
#define MAX_CWD_LEN 200
#define NET_ID "christopherhowe"

int parseInput(char* input, char args[MAX_TOKENS][MAX_TOKEN_LEN]);
void printArgs(char args[MAX_TOKENS][MAX_TOKEN_LEN], int numArgs);
void writePrompt();
void changeDirectories(char* path);
void executeCommand(char args [MAX_TOKENS][MAX_TOKEN_LEN], int numArgs);
void wrapError(const char* msg, char* err);

int main(){
    while(1){
        writePrompt();
        char input[100];
       
        scanf(" %[^\n]", input);
        printf("read input to be %s\n", input);
        
        char commandAndArgs[MAX_TOKENS][MAX_TOKEN_LEN];
        int numArgs = parseInput(input, commandAndArgs);
        printf("Parsed Command: %s\n", commandAndArgs[0]);
        printArgs(commandAndArgs, numArgs + 1);
        if (strcmp(commandAndArgs[0],"cd") == 0){
            changeDirectories(commandAndArgs[1]);
        } else if (strcmp(commandAndArgs[0],"exit") == 0){
            break;
        } 
        else {
            printf("input: %s\n",input);
            executeCommand(commandAndArgs, numArgs);
        }
    }
    return 0;
}
void wrapError(const char* msg, char* err){
    printf("SimpleShell: %s %s\n", msg, err);
}
void writePrompt(){
    char cwd[MAX_CWD_LEN];
    getcwd(cwd, MAX_CWD_LEN);
    printf("%s:%s$ ", NET_ID, cwd);
}

int parseInput(char* input, char args[MAX_TOKENS][MAX_TOKEN_LEN]){
    int numArgs = 0;
    char* token = strtok(input, " ");
    strcpy(args[0], token); // command
    token = strtok(NULL, " ");
    while (token != NULL){
        strcpy(args[numArgs + 1], token);
        token = strtok(NULL, " ");
        numArgs++;
    }
    return numArgs;
}
// TO DELETE
void printArrayOfStrings(char strings[MAX_TOKENS][MAX_TOKEN_LEN], int numStrings) {
    for (int i = 0; i < numStrings; i++) {
        printf("%s, ", strings[i]);
    }
}
// TO DETELE
void printArgs(char args[MAX_TOKENS][MAX_TOKEN_LEN], int numArgs){
    printf("Args: ");
    printArrayOfStrings(args, numArgs);
    printf("\n");
}

void changeDirectories(char* path){
    printf("Changing directory to %s\n", path);
    if (chdir(path) != 0){
        wrapError("chdir Failed:", strerror(errno));
    }
}

void executeCommand(char args [MAX_TOKENS][MAX_TOKEN_LEN], int numArgs){
    char* newArgs[numArgs + 2];
    for (int i = 0; i < numArgs + 1; i++){
        newArgs[i] = args[i];
    }
    newArgs[numArgs + 1] = NULL;

    printf("newArgs: ");
    for (int i = 0; i < numArgs + 2; i++) {
        printf("%s, ", newArgs[i]);
    }
    printf("\n");
    
    int status;
    pid_t newPid = fork();
    if (newPid == -1){
        wrapError("fork Failed:", strerror(errno));
    } else if (newPid == 0){
        // Child Process
        printf("executing command in child\n");
        if(execvp(newArgs[0], newArgs) != 0){
            wrapError("exec Failed:", strerror(errno));
        }
        exit;
    } else {
        // Parent Process
        pid_t childPid = wait(&status);
        if (status != 0) {
            printf("Child finished with error status: %d\n", status);
        }
    }
// use the fork system call and execvp system call
// should launch a new child process and replace it with a running process with the system command
}
