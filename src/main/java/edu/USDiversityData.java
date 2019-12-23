
public class USDiversityData {

    private String stname;
    private String ctyname;
    private int year;
    private int agegrp;
    private int wa_male, wa_female;
    private int ba_male, ba_female;
    private int ia_male, ia_female;
    private int aa_male, aa_female;
    private int na_male, na_female;
    private int tom_male, tom_female;

    public USDiversityData(String data_entry) {
        String otherThanQuote = " [^\"] ";
        String quotedString = String.format(" \" %s* \" ", otherThanQuote);
        String regex = String.format("(?x) "+ // enable comments, ignore white spaces
                        ",                         "+ // match a comma
                        "(?=                       "+ // start positive look ahead
                        "  (?:                     "+ //   start non-capturing group 1
                        "    %s*                   "+ //     match 'otherThanQuote' zero or more times
                        "    %s                    "+ //     match 'quotedString'
                        "  )*                      "+ //   end group 1 and repeat it zero or more times
                        "  %s*                     "+ //   match 'otherThanQuote'
                        "  $                       "+ // match the end of the string
                        ")                         ", // stop positive look ahead
                otherThanQuote, quotedString, otherThanQuote);

        String [] data = data_entry.split(regex, -1);

        try {
            this.stname = data[3];
            this.ctyname = data[4];
            this.year = Integer.valueOf(data[5]);
            this.agegrp = Integer.valueOf(data[6]);
            this.wa_male = Integer.valueOf(data[10]);
            this.wa_female = Integer.valueOf(data[11]);
            this.ba_male = Integer.valueOf(data[12]);
            this.ba_female = Integer.valueOf(data[13]);
            this.ia_male = Integer.valueOf(data[14]);
            this.ia_female = Integer.valueOf(data[15]);
            this.aa_male = Integer.valueOf(data[16]);
            this.aa_female = Integer.valueOf(data[17]);
            this.na_male = Integer.valueOf(data[18]);
            this.na_female = Integer.valueOf(data[19]);
            this.tom_male = Integer.valueOf(data[20]);
            this.tom_female = Integer.valueOf(data[21]);
        }catch(Exception e){
            System.err.println(e.toString());
        }
    }

    /** ************************************** */
    public String getstname() {
        return stname;
    }

    public void setstname(String stname)
    {
        this.stname = stname;
    }

    /** ************************************** */
    public String getctyname()
    {
        return ctyname;
    }

    public void setctyname(String ctyname)
    {
        this.ctyname = ctyname;
    }

    /** ************************************** */
    public int getyear()
    {
        return year;
    }

    public void setyear(int year)
    {
        this.year = year;
    }

    /** ************************************** */
    public int getagegrp()
    {
        return agegrp;
    }

    public void setagegrp(int agegrp)
    {
        this.agegrp = agegrp;
    }

    /** ************************************** */
    public int getwa_male()
    {
        return wa_male;
    }

    public void setwa_male(int wa_male)
    {
        this.wa_male = wa_male;
    }

    public int getwa_female()
    {
        return wa_female;
    }

    public void setwa_female(int wa_female)
    {
        this.wa_female = wa_female;
    }

    /** ************************************** */
    public int getba_male()
    {
        return ba_male;
    }

    public void setba_male(int ba_male)
    {
        this.ba_male = ba_male;
    }

    public int getba_female()
    {
        return ba_female;
    }

    public void setba_female(int ba_female)
    {
        this.ba_female = ba_female;
    }

    /** ************************************** */
    public int getia_male()
    {
        return ia_male;
    }

    public void setia_male(int ia_male)
    {
        this.ia_male = ia_male;
    }

    public int getia_female()
    {
        return ia_female;
    }

    public void setia_female(int ia_female)
    {
        this.ia_female = ia_female;
    }

    /** ************************************** */
    public int getaa_male()
    {
        return aa_male;
    }

    public void setaa_male(int aa_male)
    {
        this.aa_male = aa_male;
    }

    public int getaa_female()
    {
        return aa_female;
    }

    public void setaa_female(int aa_female)
    {
        this.aa_female = aa_female;
    }

    /** ************************************** */
    public int getna_male()
    {
        return na_male;
    }

    public void setna_male(int na_male)
    {
        this.na_male = na_male;
    }

    public int getna_female()
    {
        return na_female;
    }

    public void setna_female(int na_female)
    {
        this.na_female = na_female;
    }

    /** ************************************** */
    public int gettom_male()
    {
        return tom_male;
    }

    public void settom_male(int tom_male)
    {
        this.tom_male = tom_male;
    }

    public int gettom_female()
    {
        return tom_female;
    }

    public void settom_female(int tom_female)
    {
        this.tom_female = tom_female;
    }
}
