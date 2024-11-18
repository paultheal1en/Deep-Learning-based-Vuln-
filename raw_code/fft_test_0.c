static void fft_test(AC3MDCTContext *mdct, AVLFG *lfg)

{

    IComplex in[FN], in1[FN];

    int k, n, i;

    float sum_re, sum_im, a;



    for (i = 0; i < FN; i++) {

        in[i].re = av_lfg_get(lfg) % 65535 - 32767;

        in[i].im = av_lfg_get(lfg) % 65535 - 32767;

        in1[i]   = in[i];

    }

    fft(mdct, in, 7);



    /* do it by hand */

    for (k = 0; k < FN; k++) {

        sum_re = 0;

        sum_im = 0;

        for (n = 0; n < FN; n++) {

            a = -2 * M_PI * (n * k) / FN;

            sum_re += in1[n].re * cos(a) - in1[n].im * sin(a);

            sum_im += in1[n].re * sin(a) + in1[n].im * cos(a);

        }

        av_log(NULL, AV_LOG_DEBUG, "%3d: %6d,%6d %6.0f,%6.0f\n",

               k, in[k].re, in[k].im, sum_re / FN, sum_im / FN);

    }

}
