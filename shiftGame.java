import java.util.Scanner;
import java.util.ArrayList;

/**
 * Shift rule game where "Alice" can play any letter.
 * 
 * Ignore the temporary lack of documentation.
 * 
 * @author Benjamin Keefer
 * @version October 30th, 2024
 */
public class shiftGame {

    private int k = 2;
    private int n = 2;
    private ArrayList<ArrayList<Integer>> sequence;

    public static void main(String[] args) {
        shiftGame s = new shiftGame();
    }

    public shiftGame() {
        Scanner input = new Scanner(System.in);
        System.out.println("Choose n:");
        n = input.nextInt();
        System.out.println("Choose k:");
        k = input.nextInt();
        sequence = new ArrayList<ArrayList<Integer>>();
        ArrayList<Integer> start = new ArrayList<Integer>();
        for(int i = 0; i < n; i++) {
            start.add(0);
        }
        sequence.add(start);
        b(input);
    }

    private void b(Scanner input) {
        System.out.println("Bob plays (y/n)?");
        String yesno = input.next().toLowerCase();
        if(yesno.equals("end")) {
            input.close();
            return;
        }
        if(yesno.equals("yes") || yesno.equals("y")) {
            boolean contained = b_increment();
            if(contained) {
                System.out.println("Bob wins");
                input.close();
                return;
            }
        }
        else if(yesno.equals("no") || yesno.equals("n")) {
            boolean contained = a(input);
            if(contained) {
                input.close();
                return;
            }
        }
        else if(yesno.equals("end")) {
            input.close();
            return;
        }
        b(input);
        input.close();
    }

    private boolean a(Scanner input) {
        System.out.println("Alice plays (y/n)?");
        String yesno = input.next().toLowerCase();
        if(yesno.equals("end")) {
            return true;
        }
        if(yesno.equals("yes") || yesno.equals("y")) {
            System.out.println("Enter next value");
            int shiftToken = input.nextInt();
            if(shiftToken < 0 || shiftToken > k - 1) {
                System.out.println("Letter out of bounds");
            }
            else {
                try {
                boolean contained = a_shift(shiftToken);
                if(contained) {
                    return true;
                }
                }
                catch(Exception e) {
                    a(input);
                    return false;
                }
            }
        }
        else if(yesno.equals("no") || yesno.equals("n")) {
            boolean contained = a_shift();
            if(contained) {
                return true;
            }
        }
        else if(yesno.equals("end")) {
            input.close();
            return true;
        }
        b(input);
        input.close();
        return true;
    }

    private boolean b_increment() {
        ArrayList<Integer> nextWord = copy(sequence.getLast());
        if(!(nextWord.getFirst() < k - 1)) {
            nextWord.removeFirst();
            nextWord.add(k - 1);
        }
        else {
            nextWord.add(nextWord.removeFirst() + 1);
        }
        if(sequence.contains(nextWord)) {
            return true;
        }
        boolean end = true;
        for (Integer x : nextWord) {
            System.out.print(x);
            if(x != 0) {
                end = false;
            }
        }
        System.out.println();
        sequence.add(nextWord);
        return end;
    }

    private boolean a_shift() {
        ArrayList<Integer> nextWord = copy(sequence.getLast());
        nextWord.add(nextWord.removeFirst());
        if(sequence.contains(nextWord)) {
            System.out.println("Bob wins");
            return true;
        }
        for (Integer x : nextWord) {
            System.out.print(x);
        }
        System.out.println();
        sequence.add(nextWord);
        return false;
    }

    private boolean a_shift(int next) {
        if(next < 0 || next >= k) {
            throw new IllegalArgumentException();
        }
        ArrayList<Integer> nextWord = copy(sequence.getLast());
        nextWord.removeFirst();
        nextWord.add(next);
        if(sequence.contains(nextWord)) {
            System.out.println("Bob wins");
            return true;
        }
        for (Integer x : nextWord) {
            System.out.print(x);
        }
        System.out.println();
        sequence.add(nextWord);
        return false;
    }

    private ArrayList<Integer> copy(ArrayList<Integer> original) {
        ArrayList<Integer> copy = new ArrayList<>();
        for (Integer x : original) {
            copy.add(x);
        }
        return copy;
    }
}