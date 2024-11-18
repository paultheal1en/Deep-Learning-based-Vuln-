static void bl_intrp(EVRCContext *e, float *ex, float delay)

{

    float *f;

    int offset, i, coef_idx;

    int16_t t;



    offset = lrintf(fabs(delay));



    t = (offset - delay + 0.5) * 8.0 + 0.5;

    if (t == 8) {

        t = 0;

        offset--;

    }



    f = ex - offset - 8;



    coef_idx = t * (2 * 8 + 1);



    ex[0] = 0.0;

    for (i = 0; i < 2 * 8 + 1; i++)

        ex[0] += e->interpolation_coeffs[coef_idx + i] * f[i];

}
