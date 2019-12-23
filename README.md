# Largest triangle

## Description
This project is an implementation of the Message Passing Interface MPI in order 
to compute the largest triangle formed from a set of random points. 
Each point is represented in a planar system with x and y coordinated, 
and combinations of three points are examined to deternime the set with the largest area.

## Performance
During tests on my local machine and on RIT's cluster machine tardis.cs.rit.edu,
the sequential finishes within 85 seconds whille the cluster parallel programs
takes 6 seconds.

**Please refer to section *Syntax and output* below to understand better
the difference between the time taken by the sequential program and the parallel
programs.**

## Instructions
### Packages
- Download and install JDK 8+
- Download mpi.jar
- Generate a hostfile mentioning which machines you intend to use

### Syntax and output

On tardis.cs.rit.edu, the performance obtained are as follows.

1. Sequential version:
- `javac largest_triangle_seq.java`
- `java largest_triangle_seq 3000 100 142857`

$Output >>

           1577 0.0038473 79.791
           2489 99.531 99.740
           2924 96.984 0.14588
           4930.8
           largest_triangle_seq : 85.876 seconds

2. Parallel version:
- `mpijavac -cp /usr/local/pub/ph/mpi.jar *.java`
- `mpirun --hostfile /usr/local/pub/ph/TardisCluster3Nodes.txt --prefix /usr/local java -cp /usr/local/pub/ph/mpi.jar largest_triangle_mpi 3000 100 142857`

$Output >>

            1577 0.0038473 79.791
            2489 99.531 99.740
            2924 96.984 0.14588
            4930.8
            largest_triangle_mpi : 6.36 seconds

