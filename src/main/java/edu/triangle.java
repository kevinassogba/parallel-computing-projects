import java.util.*;

class triangle {

    Point a, b, c;
    int[] units;
    double area;
    static double max_area;

    public triangle (Point a, Point b, Point c, int[] idx)
    {
        this.a = a;
        this.b = b;
        this.c = c;
        this.units = idx;
        this.get_area();
    }

    public double get_area ()
    {
        double a = distance(this.a, this.b);
        double b = distance(this.b, this.c);
        double c = distance(this.c, this.a);

        double s = (a + b + c) / 2;
        this.area = Math.sqrt( s * ( s - a ) * ( s - b ) * ( s - c ) );
        return this.area;

    }

    public int[] get_indices()
    {
        return this.units;
    }

    public double distance(Point p1, Point p2)
    {
        double x = p1.getX() - p2.getX();
        double y = p1.getY() - p2.getY();

        return Math.sqrt( Math.pow(x,2) + Math.pow(y,2) );
    }

    public boolean is_largest()
    {
        if (this.area > max_area) {
            max_area = this.area;
            return true;
        }
        return false;
    }

    public static void reset() {
        max_area = 0;
    }

}
