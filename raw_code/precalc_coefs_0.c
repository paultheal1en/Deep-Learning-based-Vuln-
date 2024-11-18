static int16_t *precalc_coefs(double dist25, int depth)

{

    int i;

    double gamma, simil, C;

    int16_t *ct = av_malloc((512<<LUT_BITS)*sizeof(int16_t));

    if (!ct)

        return NULL;



    gamma = log(0.25) / log(1.0 - FFMIN(dist25,252.0)/255.0 - 0.00001);



    for (i = -255<<LUT_BITS; i <= 255<<LUT_BITS; i++) {

        double f = ((i<<(9-LUT_BITS)) + (1<<(8-LUT_BITS)) - 1) / 512.0; // midpoint of the bin

        simil = 1.0 - FFABS(f) / 255.0;

        C = pow(simil, gamma) * 256.0 * f;

        ct[(256<<LUT_BITS)+i] = lrint(C);

    }



    ct[0] = !!dist25;

    return ct;

}
