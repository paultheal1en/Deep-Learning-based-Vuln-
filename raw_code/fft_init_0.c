static av_cold int fft_init(AVCodecContext *avctx, AC3MDCTContext *mdct, int ln)

{

    int i, n, n2;

    float alpha;



    n  = 1 << ln;

    n2 = n >> 1;



    FF_ALLOC_OR_GOTO(avctx, mdct->costab, n2 * sizeof(*mdct->costab), fft_alloc_fail);

    FF_ALLOC_OR_GOTO(avctx, mdct->sintab, n2 * sizeof(*mdct->sintab), fft_alloc_fail);



    for (i = 0; i < n2; i++) {

        alpha     = 2.0 * M_PI * i / n;

        mdct->costab[i] = FIX15(cos(alpha));

        mdct->sintab[i] = FIX15(sin(alpha));

    }



    return 0;

fft_alloc_fail:

    mdct_end(mdct);

    return AVERROR(ENOMEM);

}
