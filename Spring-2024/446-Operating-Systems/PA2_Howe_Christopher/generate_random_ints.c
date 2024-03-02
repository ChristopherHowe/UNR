#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    // Set seed for random number generation
    srand(time(NULL));

    // Open file for writing
    FILE *file = fopen("random_numbers.txt", "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing.\n");
        return 1;
    }

    // Generate and write 100,000,000 random integers to the file
    for (int i = 0; i < 100000000; i++) {
        int random_number = rand() % 100001; // Generate random number between 0 and 100,000
        fprintf(file, "%d\n", random_number);
    }

    // Close the file
    fclose(file);

    printf("Random numbers written to file successfully.\n");

    return 0;
}
