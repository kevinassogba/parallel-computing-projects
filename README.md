# Lemoine's Conjecture verification

## Description
This repository contains codes written for a project on verifying the Lemoine's
conjecture theory. According to the theory, all odd numbers n greater that 5 are
the sum of an odd prime number p and the double of an odd/even prime number q.
In other word:

        n = p + 2 * q

Here is a [link](https://gitlab.com/SpiRITlab/parallelcomputing/tree/master/lemoine_conjecture) to more explanation on the assignment.

The repository contains 4 files (2 files for serial/sequential computing codes and
2 files for parallel computing) which are divided into 2 versions.

### Version 1
This version consists in ***reproducing a new set of prime numbers*** used for
comparison with a computed value of q. Therefore, we do not call the prime class
to get a value for q. Consequently the program runs faster.

### Version 2
This version consists in ***only calling the Prime class for both p and q***. In this case,
the program is slower.


## Performance
As mentionned above, ***version 1 performs better than version 2*** both on
the sequential and parallel algorithms. 
However during tests on my local machine and kraken, the sequential and
parallel programs of version 1 finish within 1 second, without big difference
in computation.
The most significant difference is observed with version 2, where the parallel
is significantly faster that the sequential program.

**Please refer to section *Syntax and output* below to understand better
the difference between the time taken by the sequential program and the parallel
program in both versions.**

## Instructions
### Packages
- Download and install JDK 8+
- Download omp4j jar file from [here](http://www.omp4j.org/download)

### Syntax and output
1. Sequential version 1:
- `javac lemoine_sequential_v1.java`
- `java lemoine_sequential_v1 1000001 1999999`

$Output >>
            1746551 = 1237 + 2 * 872657
            
            1.468 seconds

2. Parallel version 1:
- `java -jar <<location of the omp4j jarfile>> lemoine_omp_v1.java Prime.java`
- `java lemoine_omp_v1 1000001 1999999`

$Output >>
            1746551 = 1237 + 2 * 872657

            1.92 seconds

3. Sequential version 2:
- `javac lemoine_sequential_v2.java`
- `java lemoine_sequential_v2 1000001 1999999`

$Output >>
            1746551 = 1237 + 2 * 872657

            204.405 seconds

4. Parallel version 2:
- `java -jar <<location of the omp4j jarfile>> lemoine_omp_v2.java Prime.java`
- `java lemoine_omp_v2 1000001 1999999`

$Output >>
            1746551 = 1237 + 2 * 872657

            6.755 seconds

