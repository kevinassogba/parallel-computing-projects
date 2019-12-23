import java.util.*;

public class gencombs {
    ArrayList<int[]> combination;
    public gencombs(int numElement) {
        this.combination = new ArrayList<int[]>();
        generate(numElement);
    }

    public void generate(int numElement) {
        int[] currArr = {0,1,2};
        boolean reached = false;
        while (!reached) {
            this.combination.add(currArr);
            currArr = runComb(currArr, numElement);
            if ( currArr[0] == numElement - 3 ) {
                this.combination.add(currArr);
                reached = true;
            }
        }
        System.out.println("There are " + this.combination.size() + " combination of 3 elements in an array of " + numElement + " numbers.");
    }

    public int[] runComb(int[] currArr, int numElement) {

        if (currArr[1] == numElement - 2) {
            ++currArr[0];
            currArr[1] = currArr[0] + 1;
            currArr[2] = currArr[1] + 1;
        } else if(currArr[2] == numElement - 1) {
            ++currArr[1];
            currArr[2] = currArr[1] + 1;
        } else {
            ++currArr[2];
        }

        return currArr;
    }

    public static void main(String[] args) {

        gencombs instance = new gencombs(100);

    }


}
