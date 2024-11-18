static int expand(AVFilterContext *ctx, double *pz, int nb, double *coeffs)

{

    int i;



    coeffs[0] = 1.0;

    coeffs[1] = 0.0;



    for (i = 0; i < nb; i++) {

        coeffs[2 * (i + 1)    ] = 0.0;

        coeffs[2 * (i + 1) + 1] = 0.0;

    }



    for (i = 0; i < nb; i++)

        multiply(pz[2 * i], pz[2 * i + 1], nb, coeffs);



    for (i = 0; i < nb + 1; i++) {

        if (fabs(coeffs[2 * i + 1]) > DBL_EPSILON) {

            av_log(ctx, AV_LOG_ERROR, "coeff: %lf of z^%d is not real; poles/zeros are not complex conjugates.\n",

                   coeffs[2 * i + i], i);

            return AVERROR(EINVAL);

        }

    }



    return 0;

}
