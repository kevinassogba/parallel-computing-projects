/**
    This program is the sequential version of the lemoine conjecture verification
    algorithm.

    Version: 1

        The design of this version consists in reproducing a set of prime in
        in the range from 0 up to n/2, in order to check after computation of
        q that the resulting odd/even prime number is in the set. On that basis
        we would have found our triplet of n, p and q verifying the conjecture.

    Course: CSCI 654 - Foundations of Parallel Computing

    @author Kevin Assogba (kta7930)
*/

import java.util.Scanner;
import java.util.*;

public class lemoine_sequential_v1 {

    static Prime.Iterator valueofP = new Prime.Iterator();
    static boolean foundResult = false;
    static int upper_bound;
    static int lower_bound;
    static int[] max_value = new int[3];
    static HashSet<Integer> primeSet = new HashSet<>();

    /**
    * This function is used to find a match for a given integer such
    * as to verify its satisfaction of the Lemoine's conjecture.
    * We compute q = (n - p) / 2 and check if it is in our prime set.
    *
    *
    * @param objectiveNumber   An odd integer to address
    * @param currentValueOfP   value of p used for verification
    *
    * @return 0 if the equation does not yield the objective number
    */
    public static int findConjecture (int objectiveNumber,int currentValueOfP) {

        int currentValueOfQ = (objectiveNumber - currentValueOfP) / 2;

        // Ensure that the value of q is an integer number contained in the set
        if ( (objectiveNumber - currentValueOfP) % 2 == 0  &&
                                primeSet.contains(currentValueOfQ) ) {
            foundResult = true;
            return currentValueOfQ; // If found, return the value
        } else {
            foundResult = false;
            return 0; // Otherwise, return 0
        }
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
                valueofP.restart(); // Reset p to its first value 3 for every n
                int oddPrimeNumber;
                int oddEvenPrime;
                do {
                    oddPrimeNumber = valueofP.next(); // iterate through p
                    oddEvenPrime = findConjecture(objectiveNumber,oddPrimeNumber);
                } while(! foundResult); // As long as no result is found

                // System.out.println("\t" + objectiveNumber + " = " +
                //                  oddPrimeNumber + " + 2 * " + oddEvenPrime);

                updateMax(objectiveNumber, oddPrimeNumber, oddEvenPrime);
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
    * This method builds a set of prime number in the range of 0 to half
    * the upper bound because the set is used to verify the value of q.
    * Knowing from the equation that q is in that range, only those number
    * will serve for comparison.
    *
    */
    public static void buildPrimeSet() {

        int limit = (int) Math.floor(upper_bound/2);
        for (int number = 2; number <= limit; number++) {
            if(Prime.isPrime(number)) {
                primeSet.add(number);
            }
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
                buildPrimeSet(); // Build a set of prime numbers
                conjecture(); // Verify the lemoine conjecture
                float sec = (System.currentTimeMillis() - start) / 1000F;
                System.out.println(sec + " seconds");
        } else {

            System.out.println("Please enter valid bounds");
        }
    }
}
