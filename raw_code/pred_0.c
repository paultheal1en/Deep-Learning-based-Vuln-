static int pred(float *in, float *tgt, int n)

{

    int x, y;

    float *p1, *p2;

    double f0, f1, f2;

    float temp;



    if (in[n] == 0)

        return 0;



    if ((f0 = *in) <= 0)

        return 0;



    for (x=1 ; ; x++) {

        if (n < x)

            return 1;



        p1 = in + x;

        p2 = tgt;

        f1 = *(p1--);

        for (y=x; --y; f1 += (*(p1--))*(*(p2++)));



        p1 = tgt + x - 1;

        p2 = tgt;

        *(p1--) = f2 = -f1/f0;

        for (y=x >> 1; y--;) {

            temp = *p2 + *p1 * f2;

            *(p1--) += *p2 * f2;

            *(p2++) = temp;

        }

        if ((f0 += f1*f2) < 0)

            return 0;

    }

}
