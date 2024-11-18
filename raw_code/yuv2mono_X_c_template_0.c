yuv2mono_X_c_template(SwsContext *c, const int16_t *lumFilter,

                      const int16_t **lumSrc, int lumFilterSize,

                      const int16_t *chrFilter, const int16_t **chrUSrc,

                      const int16_t **chrVSrc, int chrFilterSize,

                      const int16_t **alpSrc, uint8_t *dest, int dstW,

                      int y, enum AVPixelFormat target)

{

    const uint8_t * const d128=dither_8x8_220[y&7];

    int i;

    unsigned acc = 0;

    int err = 0;



    for (i = 0; i < dstW; i += 2) {

        int j;

        int Y1 = 1 << 18;

        int Y2 = 1 << 18;



        for (j = 0; j < lumFilterSize; j++) {

            Y1 += lumSrc[j][i]   * lumFilter[j];

            Y2 += lumSrc[j][i+1] * lumFilter[j];

        }

        Y1 >>= 19;

        Y2 >>= 19;

        if ((Y1 | Y2) & 0x100) {

            Y1 = av_clip_uint8(Y1);

            Y2 = av_clip_uint8(Y2);

        }

        if (c->flags & SWS_ERROR_DIFFUSION) {

            Y1 += (7*err + 1*c->dither_error[0][i] + 5*c->dither_error[0][i+1] + 3*c->dither_error[0][i+2] + 8 - 256)>>4;

            c->dither_error[0][i] = err;

            acc = 2*acc + (Y1 >= 128);

            Y1 -= 220*(acc&1);



            err = Y2 + ((7*Y1 + 1*c->dither_error[0][i+1] + 5*c->dither_error[0][i+2] + 3*c->dither_error[0][i+3] + 8 - 256)>>4);

            c->dither_error[0][i+1] = Y1;

            acc = 2*acc + (err >= 128);

            err -= 220*(acc&1);

        } else {

            accumulate_bit(acc, Y1 + d128[(i + 0) & 7]);

            accumulate_bit(acc, Y2 + d128[(i + 1) & 7]);

        }

        if ((i & 7) == 6) {

            output_pixel(*dest++, acc);

        }

    }

    c->dither_error[0][i] = err;



    if (i & 6) {

        output_pixel(*dest, acc);

    }

}
