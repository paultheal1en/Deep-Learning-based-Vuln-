static int eval_lpc_coeffs(const float *in, float *tgt, int n)

{

    int x, y;

    double f0, f1, f2;



    if (in[n] == 0)

        return 0;



    if ((f0 = *in) <= 0)

        return 0;



    in--; // To avoid a -1 subtraction in the inner loop



    for (x=1; x <= n; x++) {

        f1 = in[x+1];



        for (y=0; y < x - 1; y++)

            f1 += in[x-y]*tgt[y];



        tgt[x-1] = f2 = -f1/f0;

        for (y=0; y < x >> 1; y++) {

            float temp = tgt[y] + tgt[x-y-2]*f2;

            tgt[x-y-2] += tgt[y]*f2;

            tgt[y] = temp;

        }

        if ((f0 += f1*f2) < 0)

            return 0;

    }



    return 1;

}
