/**
    This program is the sequential version of the lemoine conjecture verification
    algorithm.

    Version: 2

        The design of this version consists in reproducing iterating through
        all values of p from the Prime class as well all values of q in the
        range from 0 up to n/2.

    Course: CSCI 654 - Foundations of Parallel Computing

    @author Kevin Assogba (kta7930)
*/

import java.util.*;

public class lemoine_sequential_v2 {

    static Prime.Iterator valueofP = new Prime.Iterator();
    static boolean foundResult = false;
    static int upper_bound;
    static int lower_bound;
    static int[] max_value = new int[3];

    /**
    * This is a recursive function used to find a match for a given integer such
    * as to verify its satisfaction of the Lemoine's conjecture.
    * We compute m = p + 2*q, increment p if m < n, or decrement q if m > m
    * until a solution is found.
    *
    * @param objectiveNumber   An odd integer to address
    * @param currentValueOfP   value of p used for verification
    *
    * @return -1 if the equation does not yield the objective number
    */
    public static int[] findConjecture (int objectiveNumber, int currentValueOfP, int currentValueOfQ) {

        int[] result = {-1, currentValueOfQ};
        if ( Prime.isPrime(currentValueOfQ) && currentValueOfQ < objectiveNumber/2) {

            int currentValueOfN = currentValueOfP + 2 * currentValueOfQ;

            if ( currentValueOfN == objectiveNumber ) {

                foundResult = true;
                result[0] = 1;
                return result;
            } else if ( currentValueOfN < objectiveNumber) { //return and inc p
                result[0] = 0;
                return result;
            }
        }
        // decrement q and recursively compute
        return findConjecture(objectiveNumber, currentValueOfP, currentValueOfQ - 1);

    }

    /**
    * This method is used to iterate over all odd integers between the lower
    * and upper bounds given in command line. The method also goes through the
    * value of p. The iteration continues as long as a result is not found or
    * the value of p become larger than that of the objective integer value.
    */
    public static void conjecture() {

        for ( int objectiveNumber = lower_bound;
                    objectiveNumber <= upper_bound; objectiveNumber++ ) {

            if ( isOdd(objectiveNumber) ) {
                valueofP.restart(); // Each thread sets p to its first value 3
                foundResult = false;
                int oddPrimeNumber = 2;
                int oddEvenPrime = objectiveNumber / 2;
                //integers array containing results from findConjecture function
                int[] result = new int[2];
                do {
                    if (result[0] == 0) {
                        oddPrimeNumber = valueofP.next();
                    }
                    result = findConjecture(objectiveNumber, oddPrimeNumber, oddEvenPrime);
                } while(! foundResult );// As long as no result is found

                // System.out.println("\t" + objectiveNumber + " = " +
                //                  oddPrimeNumber + " + 2 * " + result[1]);

                updateMax(objectiveNumber, oddPrimeNumber, result[1]);
            }
        }
        System.out.println("\n" + max_value[0] + " = " +
                            max_value[1] + " + 2 * " + max_value[2] + "\n");
        System.out.println();
    }

    /**
    * This method check if an integer is odd or not.
    *
    * @param anInteger  Odd or even integer
    * @return true if aninteger is odd
    */
    public static boolean isOdd(int anInteger) {

        if (anInteger % 2 == 0) {
            return false;
        }
        return true;
    }

    /**
    * This method is used to keep track of the p values and update
    * the maximum value of p. It also tracks the corresponding value
    * of q and that of the objective number.
    *
    * @param  n  Integer to update if the input p is the maximum
    * @param  p  Integer to update if it is the maximum
    * @param  q  Integer ot update if the input q is the maximum
    */
    public static void updateMax(int n, int p, int q) {

        // n is at position 0; p is at position 1; q is at position 2
        if (p >= max_value[1]) {
            max_value[0] = n;
            max_value[1] = p;
            max_value[2] = q;
        }
    }

    /**
    * This is the main function of this class. It get the bounds
    * from the command line inputs, checks if they are odd integers
    * and calls the method to verify the conjecture.
    *
    * @param args array of Strings parsed on the command line
    */
    public static void main(String[] args) {

        lower_bound = Integer.valueOf(args[0]);
        upper_bound = Integer.valueOf(args[1]);

        if (isOdd(lower_bound) && lower_bound > 5 &&
            isOdd(upper_bound) && upper_bound >= lower_bound ) {
                long start = System.currentTimeMillis();
                conjecture(); // Verify the lemoine conjecture
                float sec = (System.currentTimeMillis() - start) / 1000F;
                System.out.println(sec + " seconds");
        } else {

            System.out.println("Please enter valid bounds");
        }
    }
}
