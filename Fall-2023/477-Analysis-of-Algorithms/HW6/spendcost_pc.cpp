/*
Author: Chris Howe
Date: November 21 2023
Title: CS 477 HW 6
*/


// Preprocessor Directives
#include <iostream>
#include <fstream>


using namespace std;
#define OUT_FILE "spendcost_pc_out.txt"

//Classes
class TestSet {
public:
    int* aSal;
    int* bSal;
    int n;
    int F, Da, Db;

    TestSet(int* newASal, int* newBSal, int newN, int newF, int newDa, int newDb)
    : aSal(newASal), bSal(newBSal), n(newN), F(newF), Da(newDa), Db(newDb) {}

    friend std::ostream& operator<<(std::ostream& os, const TestSet& data) {
        os << endl << "aSal: ";
        for (int i = 0; i < 4; ++i) {
            os << data.aSal[i] << " ";
        }

        os << endl << "bSal: ";
        for (int i = 0; i < 4; ++i) {
            os << data.bSal[i] << " ";
        }

        os << endl << "F: " << data.F << " Da: " << data.Da << " Db: " << data.Db << endl;

        return os;
    }
};

// Function Prototypes
int findLowestCost(int optSols[][2], int optChoices[][2], int*, int*, int, int, int, int);
void findLowestCost(TestSet);
void writeTableToFile(int choicesMade[][2], int);

//Main Loop
int main() {
    int aSalArray[] = {3500, 1500, 2000, 1500};
    int bSalArray[] = {2500, 1000, 3500, 2000};
    TestSet defaultTestSet = TestSet(
        aSalArray,
        bSalArray,
        4, 200, 500, 400
    );

    TestSet testSet = defaultTestSet;

    cout << defaultTestSet;
    findLowestCost(testSet);
    return 0;
}

// Dynamic programming implimentation. 
int findLowestCost(int optSols[][2], int optChoices[][2], int* aSals, int* bSals, int n, int F, int Da, int Db){
    // Fill the starting values into the table.
    optSols[0][0] = aSals[0];
    optSols[0][1] = bSals[0];

    // For each subsequent week, find the next most optimal solution.
    for(int i = 1; i < n; i++){ // starts at one since 0 is already determined
        // Determine the next val for optSols[A] (Current state is A)
        int atAPrevDayAtA = optSols[i-1][0] - Da + aSals[i];
        int atAPrevDayAtB = optSols[i-1][1] + F + aSals[i];
        if (atAPrevDayAtA <= atAPrevDayAtB){ // If it was better to be at A the previous day
            optSols[i][0] = atAPrevDayAtA;
            optChoices[i][0] = -1;
        } else { // If it was better to be at B the previous day
            optSols[i][0] = atAPrevDayAtB;
            optChoices[i][0] = 1;
        }
        
        // Determine the next val for optSols[B] (Current state is B)
        int atBPrevDayAtA = optSols[i-1][0] + F + bSals[i];
        int atBPrevDayAtB = optSols[i-1][1] - Db  + bSals[i];

        if (atBPrevDayAtA <= atBPrevDayAtB){ // If it was better to be at A the previous day
            optSols[i][1] = atBPrevDayAtA;
            optChoices[i][1] = -1;
        } else { // If it was better to be at B the previous day
            optSols[i][1] = atBPrevDayAtB;
            optChoices[i][1] = 1;
        }
    }

    // Return the final Value
    if (optSols[n-1][1] >= optSols[n-1][0]){ // If the last week being A is cheaper, return its price
        return optSols[n-1][0];
    }
    else { // Otherwise return the price of B
       return optSols[n-1][1];
    }
}


// Wrapper func for find lowest cost, creates the table and calls the func with a test case.
void findLowestCost(TestSet test){
    int optSols [test.n][2];
    int choiceMade [test.n][2]; // Stores which choice is made by the Algorithm. A=-1,B=1.
    int lowestCost = findLowestCost(optSols, choiceMade, test.aSal, test.bSal, test.n, test.F, test.Da, test.Db);
    writeTableToFile(choiceMade, test.n);
    cout << "Lowest Cost: " << lowestCost << endl;
}


// Writes the output table
void writeTableToFile(int choiceMade[][2], int n) {
    ofstream outputFile(OUT_FILE);

    if (!outputFile.is_open()) {
        cout << "Error opening the file: " << OUT_FILE << endl;
        return;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 2; ++j) {
            char outChar;
            if (choiceMade[i][j] == -1) {
                outChar = 'A';
            } else if (choiceMade[i][j] == 1){
                outChar = 'B';
            } else {
                outChar = '-';
            }

            outputFile << outChar;
            if (j < 1) {
                outputFile << "  |  ";
            }
        }
        outputFile << endl;
    }
    outputFile.close();
}


