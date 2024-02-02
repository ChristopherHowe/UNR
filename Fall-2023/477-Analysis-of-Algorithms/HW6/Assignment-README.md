# Homework 6

## Academic Honesty
**Important note for programming assignments**: As per course policies in the syllabus,
the following rules apply (check syllabus for the complete list):
* Every programming assignment must include the following statement: 
**“This code is my own work, it was written without consulting a tutor or code written by other students.”**
* Sharing ideas with other students is fine, but you should write your own code. 
Never copy or read other students’ code, including code from previous years. 
Cosmetic changes, such as rewriting comments, changing variable names and so forth – 
to disguise the fact that the work is copied from someone else is easy to detect and not allowed.
* If you find some external code (such as an open-source project) that could be re-used as part of your assignment, 
you should first contact the instructor to see whether it is fine to reuse it. 
If you decide to reuse the external code, you should clearly cite it in comments and keep the original copyright in your code, if applicable.
* It is your responsibility to keep your code private. Sharing your code in public is prohibited and may result in zero credit for the entire assignment.


## Problem one summary
**Goal:** Find the lowest cost based on the inputs
* One Delivery Truck.
* Need to decide whether to make deliveries in city A or B
* There is one warehouse in each city.
* The team of workers at each warehouse is different every week.
* For every week i, the salaries paid to the workers at each warehouse is Ai and Bi respectively.
* If you change cities between two weeks, there is a fuel cost F.
    * That is if city(i) != city(i-1) add a fuel cost.
* If you do not change cities there is a discount Da or Db dependent on the city
*   That is if city(i) == city(i-1) subtract Da/Db from week i.
* The number of weeks in a sequence is n
* A plan is a sequence of cities in n weeks.
* The plan can start in either of the two cities.
* Algorithm should take the following parameters:
    * Array of employee salaries for warehouses A and B
    * Discounts for warehouse A and B
    * Fuel cost F

### Problem Parts
* **A:**  Write a recursive formula for computing the optimal value for the sending cost. Explain whats being optimized and how you get solutions from subproblems.
    * Submit: formula and definitions.
* **B:**  Write an algorithm that computes an optimal solution based on the recurrance from part A. 
    * Should save optimal values to a file.
    * Implimented in C++
    * Run with values below
    * Submit: Source and output; output should contain optimal val table and final optimal value 
* **C:**  Update the algorithm to STORE THE DATA to reconstruct the optimal solution.
    * Store optimal solution in table in file.
    * Submit: Seperate source and output files; output should contain: table values  with additional information needs to reconstruct the optimal solution.
* **D:**  Update the algorithm to RECONSTRUCT the optimal solution from the table created in part C
    * Submit: Seperate source and output files; output should contain: the cities visited in order

## Problem 1 UnderGraduate and Graduate Required (100pts)
Suppose that you have a delivery business which can run from two warehouses in cities A
and B. You only have one delivery truck, so each week you need to choose whether to
make deliveries in city A or B. For both warehouses you are relying on daily workers, with
the teams of workers being different each week, in each city. Thus, in each week i, paying
the worker salaries in city A or B will cost you Ai or Bi . If in week i you run deliveries from
a different city from week i-1, then you will need to pay, in week i, for the fuel needed to
relocate the truck between warehouses. Assume that this fuel cost F is always the same, as
the distance between the warehouses is not changing. However, if in week i you run
deliveries from the same city as in the previous week i-1, then in week i you will get a
discount D A or D B , from the local authorities in city where you are located that week. Given
a sequence of n weeks, a plan for your business is a sequence of cities (A or B) that
indicates in which city you performed deliveries for each week. The spending cost of the
plan is the sum of salary costs paid in the selected warehouses, along with all the fuel costs
incurred whenever the truck moved between cities, and subtracting any discounts received
during the course of the plan. Your plan can begin in either of the two cities A or B.
The goal of the problem is that given a value for the fuel cost F, discounts D A and D B , and
sequences of salary costs A1 , ..., An and B1 , ..., Bn , to find a plan with minimum spending
cost. Develop a dynamic programming algorithm that finds the value and solution for an
optimal plan using the steps outlined below.

### Part A (20Pts)
Write a recursive formula for computing the optimal value for the spending
cost (i.e., define the variable that you wish to optimize and explain how a solution to
computing it can be obtained from solutions to subproblems).
Submit: the recursive formula, along with definitions and explanations on what is
computed (in a PDF file).



### Part B (30pts)
Write an algorithm that computes an optimal solution to this problem, based
on the recurrence above. The algorithm should save in an output file the optimal values for
all the subproblems as well as the optimal value for the entire problem. Implement your
algorithm in C/C++ and run it on the following values: 
F = $200, D A = $500, D B = $400.

Week 1 Week 2 Week 3 Week 4

City A salaries $3,500 $1,500 $2,000 $1,500

City B salaries $2,500 $1,000 $3,500 $2,000

Submit:
- The source file containing your algorithm (name the file spendcost_pb.c or
spendcost_pb.cpp)
- The output file created by your algorithm (name the file spendcost_pb
_out.txt), which contains:
    - The table with the optimal values to all subproblems (save the entire table)
    - The optimal value for the entire problem (indicate this on a separate line
    after the table, even if the value is found in the table above)


### Part C (20pts)
Update the algorithm you developed at point (b) to enable the reconstruction
of the optimal solution, i.e., to store the choices you made when computing the optimal
values for each subproblem in part (b). Your updated C/C++ program should store those
choices in an auxiliary table and then save that table in an output file. Include these updates
in your implementation from point (b).
Submit:
- The source file containing your algorithm (name the file spendcost_pc.c or
spendcost_pc.cpp)
- The output file created by your algorithm (name the file spendcost_pc
_out.txt), which contains the values of the table containing the additional
information (choices) needed to reconstruct the optimal solution (print the entire
table).

### Part D (30pts)
Using the additional information computed at point (c), write an algorithm
that prints the optimal solution, i.e., it prints the sequence of cities where you performed
deliveries over the course of the 4 weeks. Implement this algorithm in C/C+.
Submit:
- The source file containing your algorithm (name the file spendcost_pd.c or
spendcost_pd.cpp)
- The output file created by your algorithm (name the file spendcost_pd
_out.txt) that contains the optimal solution to the problem given by the
numerical values in part (b).


## Problem 2, Extra Credit (20pts)
Explain what is wrong with the following argument: “The
algorithm for calculating the values of m for the Matrix Multiplication problem has to fill
in the entries in just over half of an n x n table. The algorithm’s running time is therefore
Q(n 2 ).”

## Problem 3, Extra Credit (20pts)
Show how the Longest Common Subsequence algorithm discussed in class
finds an LCS of the following strings X = ⟨B, A, C, D, B, A, F⟩ and Y = ⟨A, A, B, D, F, A, G⟩.
Indicate explicitly what is the LCS that is found by the algorithm.
