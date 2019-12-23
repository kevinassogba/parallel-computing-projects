import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.commons.cli.*;
import scala.Tuple2;
import java.io.File;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.lang.Math;
import java.util.List;


public class US_diversity_index {

    public static final String OutputDirectory = "dataset/index_report";
    public static final String DatasetFile = "dataset/census_data.csv";
    public static String objective_year;
    public static String[] objective_states;

    public static boolean deleteDirectory(File directoryToBeDeleted) {
        File[] allContents = directoryToBeDeleted.listFiles();
        if (allContents != null)
            for (File file : allContents)
                deleteDirectory(file);
        return directoryToBeDeleted.delete();
    }

    public static void evaluate(SparkSession spark) {

        Dataset data = spark.read()
                .option("header", "true")
                .option("delimiter", ",")
                .option("inferSchema", "true")
                .csv(DatasetFile);

        // Encoders are created for Java beans
        Encoder<USDiversityData> diversity_encoder = Encoders.bean(USDiversityData.class);
        Dataset<USDiversityData> div_data = data.as(diversity_encoder);

        // extract necessary data
        Dataset<Row> proc_data = extract_data(div_data);

        // Pre process dataset (Group by, sort, and compute totals)
        Dataset<Row> idx_data = process_data(proc_data);

        // Compute index and write output to file
        compute_index(idx_data);

        proc_data.show();
        idx_data.show();
    }

    public static Dataset<Row> extract_data(Dataset<USDiversityData> div_data)
    {
        // selection and basic filter
        Dataset<Row> proc_data = div_data.select(
            "STNAME", "CTYNAME", "year", "agegrp", "wa_male", "wa_female",
            "ba_male", "ba_female", "ia_male", "ia_female", "aa_male", "aa_female",
            "na_male", "na_female", "tom_male", "tom_female").filter("AGEGRP = 0");

        // filter based on command line input
        if(objective_year != "all") {
            proc_data = proc_data.filter("year = " + objective_year);
        }
        if(objective_states[0] != "all") {
            Dataset<Row> temp = proc_data.filter(proc_data.col("STNAME").equalTo(objective_states[0]));

            for (int ind = 1; ind < objective_states.length; ind++) {
                temp = temp.union(proc_data.filter(proc_data.col("STNAME").equalTo(objective_states[ind])));
            }
            return temp;
        }
        return proc_data;
    }

    public static Dataset<Row> process_data(Dataset<Row> proc_data)
    {
        // Group by state and county and get the total of each variable
        Dataset<Row> agg_data = proc_data.groupBy("STNAME", "CTYNAME").sum(
                "wa_male", "wa_female", "ba_male", "ba_female", "ia_male",
                "ia_female", "aa_male", "aa_female", "na_male", "na_female",
                "tom_male", "tom_female").sort("STNAME", "CTYNAME");

        // Compute the total population and the population of each race
        agg_data = agg_data.withColumn("total", agg_data.col("sum(wa_male)")
                .plus(agg_data.col("sum(wa_female)"))
                .plus(agg_data.col("sum(ba_male)")).plus(agg_data.col("sum(ba_female)"))
                .plus(agg_data.col("sum(ia_male)")).plus(agg_data.col("sum(ia_female)"))
                .plus(agg_data.col("sum(aa_male)")).plus(agg_data.col("sum(aa_female)"))
                .plus(agg_data.col("sum(na_male)")).plus(agg_data.col("sum(na_female)"))
                .plus(agg_data.col("sum(tom_male)")).plus(agg_data.col("sum(tom_female)")))
            .withColumn("wa", agg_data.col("sum(wa_male)").plus(agg_data.col("sum(wa_female)")))
            .withColumn("ba", agg_data.col("sum(ba_male)").plus(agg_data.col("sum(ba_female)")))
            .withColumn("ia", agg_data.col("sum(ia_male)").plus(agg_data.col("sum(ia_female)")))
            .withColumn("aa", agg_data.col("sum(aa_male)").plus(agg_data.col("sum(aa_female)")))
            .withColumn("na", agg_data.col("sum(na_male)").plus(agg_data.col("sum(na_female)")))
            .withColumn("tom",agg_data.col("sum(tom_male)").plus(agg_data.col("sum(tom_female)")))
            .withColumn("state", agg_data.col("STNAME")).withColumn("county", agg_data.col("CTYNAME"));

        // Reformat the table and select filter useful information
        Dataset<Row> idx_data = agg_data.select(
            "state", "county", "wa", "ba", "ia", "aa", "na", "tom", "total");
        return idx_data;
    }

    public static void compute_index(Dataset<Row> idx_data)
    {
        // Compute the index
        JavaRDD<String[]> rdd_one = idx_data.toJavaRDD().map(rw -> {
            long agg_val = 0;
            long tot_num = rw.getLong(rw.size() - 1);
            long temp = tot_num;
            String[] output = new String[3];
            output[0] = rw.getString(0); output[1] = rw.getString(1);
            for(int iter = 2; iter < rw.size() - 1; iter++) {
                agg_val += rw.getLong(iter) * (tot_num - rw.getLong(iter));
            }
            double value = agg_val * (1 / Math.pow(temp, 2));
            output[2] = String.valueOf(value);
            return output;
        });

        // clean directory
        deleteDirectory(new File(OutputDirectory));

        // create different rdds for each states in order to have the results of
        // each state in one file.
        List<Row> uniq_states = idx_data.select("state").distinct().collectAsList();

        for(Row stat : uniq_states) {
            String a_state = stat.getString(0);
            JavaRDD<String> state_rdd = rdd_one.filter( entry -> {
                return entry[0].equals(a_state);
            }).map( an_array -> { return Arrays.toString(an_array); });

            state_rdd.coalesce(1, true).saveAsTextFile(OutputDirectory + "/" + a_state);
        }
    }

    public static void parse_arguments(String[] args)
    {
        List<String> arguments =  Arrays.asList(args);
        System.out.println("Arguments : " + Arrays.toString(args));

        if (args.length == 2 && args[0].equals("--y")) {
            objective_year = args[1];
            objective_states = new String[1]; objective_states[0] = "all";
        } else if (args.length > 1 &&
                args[0].equals("--s") &&
                ! arguments.contains("--y")) {
            objective_year = "all";
            objective_states = Arrays.copyOfRange(args, 1, args.length);
        } else if (args.length > 3 &&
                args[0].equals("--s") &&
                arguments.contains("--y")) {
            int pos = arguments.indexOf("--y");
            objective_year = args[pos + 1];
            objective_states = Arrays.copyOfRange(args, 1, pos);
        } else {
            if(arguments.indexOf("--y") < arguments.indexOf("--s")) {
                System.out.println("Error: Wrong arguments order" +
                                    " (States are expected before the year)");
                System.exit(1);
            } else{
                objective_year = "all";
                objective_states = new String[1]; objective_states[0] = "all";
            }
        }
        System.out.println("Year : " + objective_year + " | " +
                            "States : " + Arrays.toString(objective_states));
    }

    public static void main(String[] args) throws Exception
    {
        // Create a SparkConf that loads defaults from system properties and the classpath
        SparkConf sparkConf = new SparkConf();
        sparkConf.set("spark.master", "local[4]");
        sparkConf.set("spark.sql.debug.maxToStringFields", "1000");
        //Provides the Spark driver application a for easy identification in the Spark or Yarn UI
        sparkConf.setAppName("US diversity Index");

        // Creating a session to Spark. The session allows the creation of the
        // various data abstractions such as RDDs, DataFrame, and more.
        SparkSession spark = SparkSession.builder().config(sparkConf).getOrCreate();

        // Creating spark context which allows the communication with worker nodes
        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        // Start computation
        parse_arguments(args);

        evaluate(spark);

        // Stop existing spark context
        jsc.close();

        // Stop existing spark session
        spark.close();
    }
}
