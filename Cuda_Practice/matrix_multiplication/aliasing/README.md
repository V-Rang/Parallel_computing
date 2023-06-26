Comparing three different memory aliasing schemes for matrix-matrix multiplication: A x B:
1. Neutral: A x B (issue: each thread accesses a unique row of A, each of which are displaced in memory. 

2. Good: At X B (transposing A and then traversing the elements of its columns, i.e. each thread accesses a unique column of At, each of which are next to each other in
memory. This results in good memory usage as data for threads can be found with single call to DRAM that will load the values needed by the threads in the cache.)

3. Bad: A x Bt (transposing B and then traversing the elements of its rows, i.e. each thread accesses a unique row of Bt, each of which are displaced far from each other in
memory. This results in bad memory usage as data for threads cannot be found with single call to DRAM that will load the values needed by the threads in the cache.)


Timings using Nsight Compute:
1. Good: 102.55 ms
2. Neutral: 87.55 ms
3. Bad: 75.92 ms
