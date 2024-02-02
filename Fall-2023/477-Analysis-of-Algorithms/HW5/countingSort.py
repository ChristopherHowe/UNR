import random

# Create a list with unique values ranging from 0 to 100


def counting_sort(A, B, k):
    C = [0] * (k + 1)
    for j in range(len(A)):
        C[A[j]] += 1
    for i in range(1, k + 1):
        C[i] += C[i - 1]
    for j in range(len(A) - 1, -1, -1):
        B[C[A[j]] - 1] = A[j]
        C[A[j]] -= 1

def in_place_counting_sort(A, k):
    C = [0] * (k + 1)
    for j in range(len(A)):
        C[A[j]] += 1
    cInd = k
    aInd = len(A) - 1
    while aInd >= 0:
        if C[cInd] == 0:
            cInd -= 1
        else:
            C[cInd] -= 1
            A[aInd] = cInd
            aInd -= 1

def main():
    A = [5, 1, 3, 4, 2, 5, 4]
    B = [0] * len(A)  # Create an output array of the same size as A
    counting_sort(A, B, 5)
    print("After counting sort, B:", B)
    in_place_counting_sort(A,5)
    print("After in place sort, A: ", A)
    random_list = []
    for _ in range(100):
        random_list.append(random.randint(0, 100))
        
    print("random_list: ", random_list)
    in_place_counting_sort(random_list,100)
    print("random_list: ", random_list)

if __name__ == "__main__":
    main()
