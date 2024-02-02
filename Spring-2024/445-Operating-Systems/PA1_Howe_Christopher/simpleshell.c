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

#define MAX_ARGS 20
#define MAX_ARG_LEN 20
#define MAX_CWD_LEN 200
#define NET_ID "christopherhowe"

int parseInput(char* input, char* command, char args[MAX_ARGS][MAX_ARG_LEN]);
void printArgs(char args[MAX_ARGS][MAX_ARG_LEN], int numArgs);
void writePrompt();
void changeDirectories(char* path);
void executeCommand(char* command, char args [MAX_ARGS][MAX_ARG_LEN], int numArgs);


int main(){
    while(1){
        writePrompt();
        char args[MAX_ARGS][MAX_ARG_LEN];
        char input[100];
        char command[20];
        scanf(" %[^\n]", input);
        printf("read input to be %s", input);
        int numArgs = parseInput(input, command, args);
        printf("Parsed Command: %s\n", command);
        printArgs(args, numArgs);
        if (strcmp(command,"cd") == 0){
            if (numArgs >= 2){
                printf("SimpleShell: cd: too many arguments");
                continue;
            }
            changeDirectories(args[0]);

        } else {
            printf("input: %s\n",input);
            executeCommand(command, args, numArgs);
        }

    }
    
    return 0;
}

void writePrompt(){
    char cwd[MAX_CWD_LEN];
    getcwd(cwd, MAX_CWD_LEN);
    printf("%s:%s$ ", NET_ID, cwd);
}

int parseInput(char* input, char* command, char args[MAX_ARGS][MAX_ARG_LEN]){
    char* token = strtok(input, " ");
    strcpy(command, token);
    int numArgs = 0;
    token = strtok(NULL, " ");
    while (token != NULL){
        strcpy(args[numArgs], token);
        token = strtok(NULL, " ");
        numArgs++;
    }
    return numArgs;
}
// TO DELETE
void printArrayOfStrings(char strings[MAX_ARGS][MAX_ARG_LEN], int numStrings) {
    for (int i = 0; i < numStrings; i++) {
        printf("%s, ", strings[i]);
    }
}
// TO DETELE
void printArgs(char args[MAX_ARGS][MAX_ARG_LEN], int numArgs){
    printf("Args: ");
    printArrayOfStrings(args, numArgs);
    printf("\n");
}

// void makeFullPath(char* path, char* fullPath){
//     if (path[0] != '/'){
//         if (path[0] == '.'){
//             if (path[1] == '.' || path[1] == ' '){
//                 fullPath = path;
//             } else {
//                 char* pathNoDot;
//                 strcpy(pathNoDot, path + 1);
//                 fullPath = strcat(getcwd(NULL, MAX_CWD_LEN),pathNoDot);
//             }
//         } else {
//             fullPath = strcat(getcwd(NULL, MAX_CWD_LEN),path);
//         }
//     } else {
//         fullPath = path;
//     }
// }

void changeDirectories(char* path){
    // char* fullPath;
    // makeFullPath(path, fullPath);
    printf("Changing directory to %s\n", path);
    if (chdir(path) != 0){
        printf("Failed to change directory to %s\n", path);
    }
}

void executeCommand(char* command, char args [MAX_ARGS][MAX_ARG_LEN], int numArgs){
    char* newArgs[numArgs + 2];
    newArgs[0] = command;
    for (int i = 1; i < numArgs + 1; i++){
        newArgs[i] = args[i - 1];
    }
    newArgs[numArgs + 1] = NULL;

    printf("newArgs: ");
    for (int i = 0; i < numArgs + 2; i++) {
        printf("%s, ", newArgs[i]);
    }
    printf("\n");
    
    pid_t newPid = fork();
    if (newPid == 0){
        printf("Hello from child\n");
        printf("executing command in child");
        execvp(command, newArgs);
    } else {
        printf("Hello from Parent\n");
    }
// use the fork system call and execvp system call
// should launch a new child process and replace it with a running process with the system command
}
