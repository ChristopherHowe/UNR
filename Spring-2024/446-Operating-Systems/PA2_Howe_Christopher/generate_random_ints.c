#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    // Check if the number of command line arguments is correct
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_of_numbers>\n", argv[0]);
        return 1;
    }

    // Convert the command line argument to an integer
    int num_of_numbers = atoi(argv[1]);

    // Set seed for random number generation
    srand(time(NULL));

    // Create the file name based on the specified number of numbers
    char file_name[50];
    sprintf(file_name, "%d-random-numbers.txt", num_of_numbers);

    // Open file for writing
    FILE *file = fopen(file_name, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        return 1;
    }

    // Generate and write the specified number of random integers to the file
    for (int i = 0; i < num_of_numbers; i++) {
        int random_number = rand() % 100001; // Generate random number between 0 and 100,000
        fprintf(file, "%d\n", random_number);
    }

    // Close the file
    fclose(file);

    printf("Random numbers written to file %s successfully.\n", file_name);

    return 0;
}
