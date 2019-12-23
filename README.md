<h1>U.S. Census Diversity Index</h1>

## Objective
<p><a name="system"></a></p>
<p>The United States Census Bureau (USCB) estimates the number of persons in each county in each state, categorized by age, gender, race, and other factors. USCB uses six racial categories: White; Black or African American; American Indian or Alaska Native; Asian; Native Hawaiian or Other Pacific Islander; Two or more races.</p>
<p>For example, here are the USCB's population estimates for Monroe County, New York, as of July 1, 2017:</p>
<p></p>
<table border="0" cellpadding="0" cellspacing="0">
<tbody>
<tr>
<td align="left" valign="top">White</td>
<td width="40"></td>
<td align="right" valign="top">573928</td>
</tr>
<tr>
<td align="left" valign="top">Black or African American</td>
<td width="40"></td>
<td align="right" valign="top">121423</td>
</tr>
<tr>
<td align="left" valign="top">American Indian or Alaska Native</td>
<td width="40"></td>
<td align="right" valign="top">3074</td>
</tr>
<tr>
<td align="left" valign="top">Asian</td>
<td width="40"></td>
<td align="right" valign="top">29053</td>
</tr>
<tr>
<td align="left" valign="top">Native Hawaiian or Other Pacific Islander</td>
<td width="40"></td>
<td align="right" valign="top">517</td>
</tr>
<tr>
<td align="left" valign="top">Two or more races</td>
<td width="40"></td>
<td align="right" valign="top">19667</td>
</tr>
<tr>
<td align="left" valign="top">Total</td>
<td width="40"></td>
<td align="right" valign="top">747662</td>
</tr>
</tbody>
</table>
<p></p>
<p>The&nbsp;<strong>diversity index</strong>&nbsp;<em>D</em>&nbsp;for a population is the probability that two randomly chosen individuals in that population will be of different races. The diversity index is calculated with this formula, where&nbsp;<em>N</em><sub><em>i</em></sub>&nbsp;is the number of individuals in racial category&nbsp;<em>i</em>&nbsp;and&nbsp;<em>T</em>&nbsp;is the total number of individuals:</p>
<tr>
<td align="right" valign="center"><em>D</em>&nbsp;&nbsp;=&nbsp;&nbsp;1/<em>T</em><sup>2</sup>&nbsp;</td>
<td align="center" valign="center">&Sigma;</td>
<td align="left" valign="center">&nbsp;<em>N</em><sub><em>i</em></sub>&nbsp;(<em>T</em>&nbsp;&minus;&nbsp;<em>N</em><sub><em>i</em></sub>)</td>
</tr>


#### Note:
Please download the [census dataset](https://www2.census.gov/programs-surveys/popest/datasets/2010-2017/counties/asrh/cc-est2017-alldata.csv) from the USCB web site and place this dataset into the ```dataset``` folder and rename the file ```census_data.csv```.
The USCB web site has a [document](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2017/cc-est2017-alldata.pdf) that describes the census dataset in full detail.

## Experiments
Aiming at designing an algorithm to achieve a fast mining of the dataset, a map-reduce parallel program using Apache Spark is implemented.
Note that the program can also be run on a cluster of computers. Taking commands such as the year and/or state or no arguments from the CLI,
this program writes the results back to file in ascending order of states and counties. 

The solution can be found in the `index_report` folder. In this folder will be generated subfolders named after the states taken into account
for the analysis. The results for a given state are collected into a single file located in its subfolder.

## Syntax and outputs
To run the code `cd` in this folder, and compile using `maven`.
In addition to writing the solution to file, two tables are generated. 
The first represents the first 20 lines of the original data, and 
the second table represents the first 20 lines of the processes data prior to computing the index.
The processing consists in extracting relevant data, sorting by states and counties and computing the total population as well as the population of each race. 


#### Computation of the index for all states in all years

* `mvn -U compile`
* `mvn package`
* `java -cp target/us_census_diversity_index-1.0-SNAPSHOT.jar US_diversity_index`

<img src="/images/raw_data_allYear_allStates.png" alt="Raw data for all states in all years" width="2000" height="400"/>

*Fig 1. Raw data for all states in all years*

![Processed data for all states in all years](/images/proc_data_allYear_allStates.png)

*Fig 2. Processed data for all states in all years*

#### Computation of the index for all states as of July 1, 2017
* `mvn -U compile`
* `mvn package`
* `java -cp target/us_census_diversity_index-1.0-SNAPSHOT.jar US_diversity_index --y 10`

<img src="/images/raw_data_Year10_allStates.png" alt="Raw data for all states as of July 1, 2017" width="2000" height="400"/>

*Fig 1. Raw data for all states as of July 1, 2017*

![Processed data for all states as of July 1, 2017](/images/proc_data_Year10_allStates.png)

*Fig 2. Processed data for all states as of July 1, 2017*

#### Computation of the index for Arkansas, Florida and Georgia states as of July 1, 2017
* `mvn -U compile`
* `mvn package`
* `java -cp target/us_census_diversity_index-1.0-SNAPSHOT.jar US_diversity_index --s Florida Georgia Arkansas --y 10`

<img src="/images/raw_data_Year10_3States.png" alt="Raw data for Arkansas, Florida and Georgia as of July 1, 2017" width="2000" height="400"/>

*Fig 1. Raw data for Arkansas, Florida and Georgia as of July 1, 2017*

![Processed data for Arkansas, Florida and Georgia as of July 1, 2017](/images/proc_data_Year10_3States.png)

*Fig 2. Processed data for Arkansas, Florida and Georgia as of July 1, 2017*
