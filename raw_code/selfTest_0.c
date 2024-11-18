static void selfTest(uint8_t *ref[4], int refStride[4], int w, int h)

{

    const int flags[] = { SWS_FAST_BILINEAR,

                          SWS_BILINEAR, SWS_BICUBIC,

                          SWS_X       , SWS_POINT  , SWS_AREA, 0 };

    const int srcW = w;

    const int srcH = h;

    const int dstW[] = { srcW - srcW/3, srcW, srcW + srcW/3, 0 };

    const int dstH[] = { srcH - srcH/3, srcH, srcH + srcH/3, 0 };

    enum PixelFormat srcFormat, dstFormat;



    for (srcFormat = 0; srcFormat < PIX_FMT_NB; srcFormat++) {

        for (dstFormat = 0; dstFormat < PIX_FMT_NB; dstFormat++) {

            int i, j, k;

            int res = 0;



            printf("%s -> %s\n",

                   sws_format_name(srcFormat),

                   sws_format_name(dstFormat));

            fflush(stdout);



            for (i = 0; dstW[i] && !res; i++)

                for (j = 0; dstH[j] && !res; j++)

                    for (k = 0; flags[k] && !res; k++)

                        res = doTest(ref, refStride, w, h, srcFormat, dstFormat,

                                     srcW, srcH, dstW[i], dstH[j], flags[k]);

        }

    }

}
