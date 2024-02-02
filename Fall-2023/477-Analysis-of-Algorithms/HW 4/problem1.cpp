#include <iostream>
#include <stdarg.h>
#include <cstring> // Include the cstring header
#include <time.h>

using namespace std;

string arrToStr(const int* arr, int size) {
    string result;
    for (int i = 0; i < size; ++i) {
        result += to_string(arr[i]); 
        result += ", ";
    }
    return result;
}

void recPrint(int depth, const char* format, ...) {
    for (int i = 0; i <= depth; i++) {
        std::cout << "   ";
    }
    std::cout << depth << ". ";

    va_list args;
    va_start(args, format);
    vfprintf(stdout, format, args);
    va_end(args);

    std::cout << std::endl;
}

void swap(int* A, int indA, int indB,int depth){
    int temp = A[indA];
    A[indA] = A[indB];
    A[indB]= temp;
}

void arrayReverse(int* A, int size, int p,int q, int depth){
    recPrint(depth,"calling arrayReverse");
    //base case
    if (p == q){
        return;
    } 
    // divide
    int mid = (p + q)/2;
    arrayReverse(A, size, p, mid, depth + 1);
    arrayReverse(A, size, mid + 1, q, depth + 1);

    // combine
    for (int i = p; i <= mid; i++){
        swap(A, i, (i + (q - p)/2 + 1), depth);
    }
    recPrint(depth, "after combine A: %s", arrToStr(A,size).c_str());
}

int main() {
    int arrSize = 8;
    int A[arrSize] = {1,2,3,4,5,6,7,8};
    arrayReverse(A,arrSize, 0, arrSize - 1, 0);
}
