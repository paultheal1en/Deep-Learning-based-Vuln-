static int process_tns_coeffs(TemporalNoiseShaping *tns, float *tns_coefs_raw,

                              int order, int w, int filt)

{

    int i, j;

    int *idx = tns->coef_idx[w][filt];

    float *lpc = tns->coef[w][filt];

    const int iqfac_p = ((1 << (MAX_LPC_PRECISION-1)) - 0.5)/(M_PI/2.0);

    const int iqfac_m = ((1 << (MAX_LPC_PRECISION-1)) + 0.5)/(M_PI/2.0);

    float temp[TNS_MAX_ORDER] = {0.0f}, out[TNS_MAX_ORDER] = {0.0f};



    /* Quantization */

    for (i = 0; i < order; i++) {

        idx[i] = ceilf(asin(tns_coefs_raw[i])*((tns_coefs_raw[i] >= 0) ? iqfac_p : iqfac_m));

        lpc[i] = 2*sin(idx[i]/((idx[i] >= 0) ? iqfac_p : iqfac_m));

    }



    /* Trim any coeff less than 0.1f from the end */

    for (i = order; i > -1; i--) {

        lpc[i] = (fabs(lpc[i]) > 0.1f) ? lpc[i] : 0.0f;

        if (lpc[i] != 0.0 ) {

            order = i;

            break;

        }

    }



    if (!order)

        return 0;



    /* Step up procedure, convert to LPC coeffs */

    out[0] = 1.0f;

    for (i = 1; i <= order; i++) {

        for (j = 1; j < i; j++) {

            temp[j] = out[j] + lpc[i]*out[i-j];

        }

        for (j = 1; j <= i; j++) {

            out[j] = temp[j];

        }

        out[i] = lpc[i-1];

    }

    memcpy(lpc, out, TNS_MAX_ORDER*sizeof(float));



    return order;

}
