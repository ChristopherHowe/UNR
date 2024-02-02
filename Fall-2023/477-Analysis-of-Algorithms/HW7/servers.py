import numpy as np

# Greedy algorithm that serves the quickest jobs first to produce the lowest total time.
def findLeastTime(jobs, s):
    sorted_jobs = sorted(jobs)
    
    totalTime = 0
    serverOccupiedTimes = np.zeros(s)
    for job in sorted_jobs: 
        jobWaited, nextAvailableServer = getNextReadyServer(serverOccupiedTimes)
        serverOccupiedTimes[nextAvailableServer] += job
        totalTime += job + jobWaited
    return totalTime

# Helper function that gets the next ready server
def getNextReadyServer(times):
    nextServer = -1
    nextServerTime = 100000000
    for serverNum, serverOpens in enumerate(times):
        if serverOpens < nextServerTime:
            nextServerTime = serverOpens
            nextServer = serverNum
    if nextServer == -1:
        raise Exception("Failed to get the index of the next ready server") 
    return nextServerTime, nextServer
     
                

def main():
    # Company has s identical servers
    # There are n customers with jobs to run on the servers.
    # Each customer i has a job that takes ti time
    # Total time a job takes is t + ti where t is the time a job waits to be served.
    # Goal is to minimize the total of the total times for all the jobs.
    numServers = 3
    customerJobs = [7, 3, 10, 5]
    print(f"least total time: ", findLeastTime(customerJobs, numServers))

if __name__ == "__main__":
    main()
