static void mdct_test(AC3MDCTContext *mdct, AVLFG *lfg)

{

    int16_t input[MDCT_SAMPLES];

    int32_t output[AC3_MAX_COEFS];

    float input1[MDCT_SAMPLES];

    float output1[AC3_MAX_COEFS];

    float s, a, err, e, emax;

    int i, k, n;



    for (i = 0; i < MDCT_SAMPLES; i++) {

        input[i]  = (av_lfg_get(lfg) % 65535 - 32767) * 9 / 10;

        input1[i] = input[i];

    }



    mdct512(mdct, output, input);



    /* do it by hand */

    for (k = 0; k < AC3_MAX_COEFS; k++) {

        s = 0;

        for (n = 0; n < MDCT_SAMPLES; n++) {

            a = (2*M_PI*(2*n+1+MDCT_SAMPLES/2)*(2*k+1) / (4 * MDCT_SAMPLES));

            s += input1[n] * cos(a);

        }

        output1[k] = -2 * s / MDCT_SAMPLES;

    }



    err  = 0;

    emax = 0;

    for (i = 0; i < AC3_MAX_COEFS; i++) {

        av_log(NULL, AV_LOG_DEBUG, "%3d: %7d %7.0f\n", i, output[i], output1[i]);

        e = output[i] - output1[i];

        if (e > emax)

            emax = e;

        err += e * e;

    }

    av_log(NULL, AV_LOG_DEBUG, "err2=%f emax=%f\n", err / AC3_MAX_COEFS, emax);

}
