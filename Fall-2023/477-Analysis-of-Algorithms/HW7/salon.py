# Chris Howe
# HW 7 Salon Problem

# Greedy algorithm that serves the customers with the longest styling times first
def getLowestCompletionTime(customers):
    sorted_customers = sorted(customers, key=lambda x: x['s'], reverse=True)
    nextStartTime = 0
    completionTime = 0
    for customer in sorted_customers:
        serviceTime = customer['w'] + customer['s']
        jobDone = nextStartTime + serviceTime
        if jobDone >= completionTime:
            completionTime = jobDone
        nextStartTime += customer['w']
    return completionTime

def main():
    # w = wash time, only one customer can have their hair washed at a time
    # s = style time, all customers can be styled at the same time, overlapping here is OK
    # Optimizing earliest/smallest completion time, completion time is the time when all customers are serviced
    customers = [
        {'w': 10, 's': 20},
        {'w': 15, 's': 25},
        {'w': 8, 's': 18},
        {'w': 9, 's': 1},
        {'w': 13, 's': 3},
        {'w': 1, 's': 9},
        {'w': 6, 's': 8},
        {'w': 2, 's': 12},
    ]

    print(f"lowest completion time: {getLowestCompletionTime(customers)}")


if __name__ == "__main__":
    main()
