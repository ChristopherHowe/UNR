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

#define MAX_TOKENS 20
#define MAX_TOKEN_LEN 200
#define MAX_CWD_LEN 200
#define NET_ID "christopherhowe"

// Function Prototypes
int parseInput(char* input, char args[MAX_TOKENS][MAX_TOKEN_LEN]);
void writePrompt();
void changeDirectories(char* path, int numArgs);
void executeCommand(char args [MAX_TOKENS][MAX_TOKEN_LEN], int numArgs);
void wrapError(const char* msg,const char* err);

// Main Loop
int main(){
    while(1){
        writePrompt();
        
        char input[100];
        scanf(" %[^\n]", input);
        
        char commandAndArgs[MAX_TOKENS][MAX_TOKEN_LEN];
        int numArgs = parseInput(input, commandAndArgs);
        if (strcmp(commandAndArgs[0],"cd") == 0){
            changeDirectories(commandAndArgs[1], numArgs);
        } else if (strcmp(commandAndArgs[0],"exit") == 0){
            break;
        } 
        else {
            executeCommand(commandAndArgs, numArgs);
        }
    }
    return 0;
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

void changeDirectories(char* path, int numArgs){
    if (numArgs != 1){
        wrapError("Path Not Formatted Correctly!","");
        return;
    }
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
    
    pid_t newPid = fork();
    if (newPid == -1){ // Edge Case
        wrapError("fork Failed:", strerror(errno));
    } else if (newPid == 0){ // Child Process
        if(execvp(newArgs[0], newArgs) != 0){
            wrapError("exec Failed:", strerror(errno));
            _exit(1);
        }
    } else { // Parent Process
        int s;
        wait(&s);
        int status = WEXITSTATUS(s); // convert Wait exit status to normal status's, c multiplies exit codes by 256.
        if (status != 0) {
            printf("Child finished with error status: %d\n", status);
        }
    }
}

void wrapError(const char* msg,const char* err){
    printf("SimpleShell: %s %s\n", msg, err);
}
