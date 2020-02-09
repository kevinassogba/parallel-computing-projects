import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Random;
import java.util.*;
import java.io.*;
import java.lang.Math;

import mpi.MPI;
import mpi.MPIException;

@SuppressWarnings("ConstantConditions")
public class opencbs {

    /**
     * convert byte[] to hex string
     *
     * @param hash the input hash
     * @return hex string
     */
    private static String bytesToHex(byte[] hash) {
        StringBuilder hexString = new StringBuilder();

        for (byte b : hash) {
            String hex = Integer.toHexString(0xff & b);
            if (hex.length() == 1) hexString.append('0');
            hexString.append(hex);
        }
        return hexString.toString();
    }

    /**
     * get a randomized target hash
     * @return randomized target hash
     */
    public static String getTargetHash(int size) {
        Random rand = new Random();
        int randInt = rand.nextInt(size);
        return SHA256(String.valueOf(randInt));
    }

    /**
     * get a sha256 of the input string
     *
     * @param inputString the input String
     * @return resulting hash in hex string
     */
    public static String SHA256(String inputString) {
        try {
            MessageDigest sha256 = MessageDigest.getInstance("SHA-256");
            return bytesToHex(sha256.digest(inputString.getBytes(
                    StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException ex) {
            System.err.println(ex.toString());
            return null;
        }
    }

    /**
     * perform the proof-of-work
     *
     * @param blockHash  hash of the blockinfo
     * @param targetHash target hash
     * @return nonce (a 32-bit integer) that satisfies the requirements
     */
    public static int[] pow(String blockHash, String targetHash, long start, long limit,
                            int size) {
        int[] nonce = new int[1];
        nonce[0] = Integer.MAX_VALUE;
        String tmp_hash;
        for (int nce = (int) start; nce <= limit; nce += size) {
            tmp_hash = SHA256(SHA256(blockHash + nce));
            if (targetHash.compareTo(tmp_hash) > 0) {
                nonce[0] = nce;
                break;
            }
        }
        return nonce;
    }

    public int[] helper(String blockHash, String targetHash, int rank, int size,
                        int chunk_size) throws MPIException {
        long start = Integer.MIN_VALUE;
        int[] nonce_found = new int[1];
        nonce_found[0] = Integer.MAX_VALUE;
        long limit = Integer.MIN_VALUE + chunk_size;
        while (limit < Integer.MAX_VALUE && nonce_found[0] == Integer.MAX_VALUE) {
            int[] ind_nonce = pow(blockHash, targetHash, start + rank, limit, size);

            // Gather all partial solutions
            int[] nonces = new int[size];
            MPI.COMM_WORLD.gather(ind_nonce, 1, MPI.INT, nonces, 1, MPI.INT, 0);

            nonce_found = found_result(nonces, rank, size, nonce_found);

            start = limit;
            limit += chunk_size;
        }
        return nonce_found;
    }

    public int[] found_result(int[] nonces, int rank, int size, int[] nonce_found)
            throws MPIException {
        if (rank == 0) {
            for (int nonce : nonces) {
                if (nonce != Integer.MAX_VALUE) {
                    nonce_found[0] = nonce;
                    break;
                }
            }
        }
        return my_bcast(nonce_found, rank, size);
    }

    public int[] my_bcast(int[] nonce_found, int rank, int size) throws MPIException {
        if (rank == 0) {
            for (int idx = 0; idx < size; idx++) {
                if (idx != rank) {
                    MPI.COMM_WORLD.send(nonce_found, 1, MPI.INT, idx, 0);
                }
            }
        } else {
            MPI.COMM_WORLD.recv(nonce_found, 1, MPI.INT, 0, MPI.ANY_TAG);
        }
        return nonce_found;
    }

    public static String HexValueDivideBy(String hexValue, int val) {
        BigInteger tmp = new BigInteger(hexValue, 16);
        tmp = tmp.divide(BigInteger.valueOf(val));
        StringBuilder newHex = new StringBuilder(bytesToHex(tmp.toByteArray()));
        while (newHex.length() < hexValue.length()) {
            newHex.insert(0, '0');
        }
        return newHex.toString();
    }

    public static String HexValueMultipleBy(String hexValue, int val) {
        BigInteger tmp = new BigInteger(hexValue, 16);
        tmp = tmp.multiply(BigInteger.valueOf(val));
        StringBuilder newHex = new StringBuilder(bytesToHex(tmp.toByteArray()));
        while (newHex.length() < hexValue.length()) {
            newHex.insert(0, '0');
        }
        return newHex.toString();
    }

    public static String get_features(String block, String target, int size)
    {
        int zeros = 0;
        String len_tar = String.valueOf(target.length());
        String blk_vs_tar = String.valueOf(block.compareTo(target));
        String com_sze = String.valueOf(size);
        StringBuilder tar = new StringBuilder(target);
        while(tar.charAt(0) == '0') {
            zeros++;
            tar.deleteCharAt(0);
        }
        String lead_zero = String.valueOf(zeros);
        String params = lead_zero +" "+ len_tar +" "+ blk_vs_tar +" "+ com_sze;
        return params;
    }

    public static int get_chunk_size(String block, String target, int size)
    {
        String chunk_size = " ";
        try {
            String params = get_features(block, target, size);
            String command = "python predict_chunk.py " + params;
            Process python_process = Runtime.getRuntime().exec(command);
            BufferedReader result = new BufferedReader(
                        new InputStreamReader(python_process.getInputStream()));
            chunk_size = result.readLine();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
        if(chunk_size == null) {
            return (int) Math.pow(10, 6);
        } else {
            return Integer.valueOf(chunk_size);
        }
    }

    public static void main(String[] args) throws MPIException {
        MPI.Init(args);
        // number of blocks to be generated or number of rounds; default to 5
        int numberOfBlocks = 10;
        // average block generation time, default to 30 Secs.
        double avgBlockGenerationTimeInSec = 30.0;
        // init block hash
        String initBlockHash = SHA256("CSCI-654 Foundations of Parallel Computing");
        // init target hash
        String initTargetHash = "0000092a6893b712892a41e8438e3ff2242a68747105de0395826f60b38d88dc";
        // com world parameters
        int rank = MPI.COMM_WORLD.getRank();
        int size = MPI.COMM_WORLD.getSize();

        int currentBlockID = 1;
        String tmpBlockHash = initBlockHash;
        String tmpTargetHash = initTargetHash;
        MyTimer myTimer;
        opencbs instance = new opencbs();
        int[] nonce;

        while (currentBlockID <= numberOfBlocks) {
            myTimer = new MyTimer("CurrentBlockID:" + currentBlockID);
            myTimer.start_timer();
            // compute the chunk size
            int chunk_size;
            if(args.length > 0 && args[0].equals("--default")) {
                chunk_size = (int) Math.pow(10, 6);
            } else {
                chunk_size = get_chunk_size(tmpBlockHash, tmpTargetHash, size);
            }
            // assign jobs to worker
            nonce = instance.helper(tmpBlockHash, tmpTargetHash, rank, size, chunk_size);
            myTimer.stop_timer();
            if (rank == 0) {
                myTimer.print_elapsed_time();
                System.out.println("Nonce:" + nonce[0]);
            }
            // found a new block
            tmpBlockHash = SHA256(tmpBlockHash + "|" + nonce[0]);

            // update the target
            if (myTimer.get_elapsed_time_in_sec() < avgBlockGenerationTimeInSec)
                tmpTargetHash = HexValueDivideBy(tmpTargetHash, 2);
            else
                tmpTargetHash = HexValueMultipleBy(tmpTargetHash, 2);

            if (rank == 0) {
                System.out.println("New Block Hash:  " + tmpBlockHash);
                System.out.println("New Target Hash: " + tmpTargetHash);
                System.out.println();
            }
            currentBlockID++;
        }
        MPI.Finalize();
    }
}
