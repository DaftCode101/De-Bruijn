import java.util.ArrayList;

/**
 * Java implementation of combinatoric "game" for De Bruijn sequence construction.
 * Paper: De Bruijn Sequences: From Games to Shift-Rules to a Proof of the Fredericksen-
 * Kessler-Mairoana Theroem.
 *
 * There's still optimization things that could be done to this code, but its fast enough for right now.
 * 
 * Translated to java by:
 * @author Benjamin Keefer
 * @version September 22, 2024
 */
public class revPMax {
    // Number of letters in alphabet
    private int k;
    // Length of words in sequence
    private int n;
    // Number of words to print after sequence is generated
    private int wordsToPrint;

    /**
     * Constructor for De Bruijn Sequence
     * @param k
     * @param n
     */
    public revPMax(int k, int n) {
        this.k = k;
        this.n = n;
        wordsToPrint = 8;
    }

    /**
     * Generates the sequence by repeatedly calling the shift function.
     */
    public void run() {
        ArrayList<ArrayList<Integer>> sequence = new ArrayList<>();
        ArrayList<Integer> s = new ArrayList<>();
        long time = System.currentTimeMillis();
        for(int i = 0; i < n; i++) {
            s.add(0);
        }
        for(int i = 0; i < Math.pow(k, n); i++) {
            sequence.addLast(s);
            s = shift(s);
        }
        long after = System.currentTimeMillis();
        printSequence(sequence);
        System.out.println("Generation time: " + (after - time) + " milliseconds.");
        printDivider();
        System.out.println("k: " + k + "  n: " + n);
    }

    /**
     * @param s
     * @return A copy of the given ArrayList
     */
    private ArrayList<Integer> copy(ArrayList<Integer> s) {
        ArrayList<Integer> copy = new ArrayList<>();
        for(int val : s) {
            copy.addLast(val);
        }
        return copy;
    }

    /**
     * Calculates the base-k value of the given word.
     * @param s
     * @return integer value
     */
    private int val(ArrayList<Integer> s) {
        int n = s.size();
        int value = 0;
        for(int i = 1; i < n + 1; i++) {
            value += s.get(n - i) * (int)Math.pow(k, i-1);
        }
        return value;
    }

    /**
     * @param s
     * @return Max value of the given word across its rotations
     */
    private int valStar(ArrayList<Integer> s) {
        int n = s.size();
        ArrayList<Integer> values = new ArrayList<>();
        values.add(val(s));
        ArrayList<Integer> x = copy(s);
        for(int i = 0; i < n - 1; i++) {
            x.add(x.removeFirst());
            values.add(val(x));
        }
        int max = 0;
        for(int val : values) {
            if(val > max) {
                max = val;
            }
        }
        return max;
    }

    /**
     * Recursively shifts the given word until the first letter is not zero.
     * @param s
     * @param len
     */
    private void rotateZeros(ArrayList<Integer> s, int len) {
        if(len == 0) {
            return;
        }
        else if(s.getFirst() == 0) {
            s.add(s.removeFirst());
            rotateZeros(s, len - 1);
        }
    }

    /**
     * Only optimal strategy for "alice".
     * @param s
     * @return boolean
     */
    private boolean aDagger(ArrayList<Integer> s) {
        if(s.getLast() == 0) {
            return false;
        }
        ArrayList<Integer> x = copy(s);
        int value = valStar(x);
        rotateZeros(x, n);
        return value == val(x);
    }

    /**
     * Returns a new word according to the shift rule.
     * @param s
     * @return
     */
    private ArrayList<Integer> shift(ArrayList<Integer> s) {
        ArrayList<Integer> x = copy(s);
        int last = x.removeLast();
        if(last < k - 1) {
            x.add(last + 1);
            if(aDagger(x)) {
                x.removeLast();
                x.addFirst(last + 1);
                return x;
            }
            x.removeLast();
        }
        x.add(last);
        if(aDagger(s)) {
            x.removeLast();
            x.addFirst(0);
            return x;
        }
        x.removeLast();
        x.addFirst(last);
        return x;
    }

    /**
     * Prints the first few words in the sequence, the middle few, and the last few.
     * Modify the constructor to change the number of words to be printed.
     * 
     * @param sequence to be printed
     */
    private void printSequence(ArrayList<ArrayList<Integer>> sequence) {
        int size = sequence.size();
        if(size > wordsToPrint * 4) {
            printDivider();
            for(int i = 0; i < wordsToPrint; i++) {
                printLine(sequence.get(i));
            }
            printDivider();
            for(int i = (int)(size / 2 - (wordsToPrint / 2)); i < (int)(size / 2 + wordsToPrint / 2); i++) {
                printLine(sequence.get(i));
            }
            printDivider();
            for(int i = size - wordsToPrint; i < size; i++) {
                printLine(sequence.get(i));
            }
            printDivider();
        }
        else {
            printDivider();
            for(int i = 0; i < sequence.size(); i++) {
                printLine(sequence.get(i));
            }
            printDivider();
        }
        System.out.println("# of words in sequence: " + sequence.size());
    }

    /**
     * Prints a word from the sequence.
     * @param s
     */
    private void printLine(ArrayList<Integer> s) {
        String sequenceRep = "";
        for(int j = 0; j < s.size() - 1; j++) {
            sequenceRep += s.get(j) + ", ";
        }
        sequenceRep += s.getLast();
        System.out.println(sequenceRep);
    }

    /**
     * Used for making output look better.
     */
    private void printDivider() {
        // System.out.println("====================================================");
        System.out.println("=======================");
        // System.out.println();
    }

    /**
     * Main method :P
     * @param args
     */
    public static void main(String[] args) {
        // Adjust k & n values here
        revPMax x = new revPMax(10, 5);
        x.run();
    }
}
