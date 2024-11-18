static av_always_inline void hyscale(SwsContext *c, int16_t *dst, int dstWidth,

                                     const uint8_t *src_in[4], int srcW, int xInc,

                                     const int16_t *hLumFilter,

                                     const int16_t *hLumFilterPos, int hLumFilterSize,

                                     uint8_t *formatConvBuffer,

                                     uint32_t *pal, int isAlpha)

{

    void (*toYV12)(uint8_t *, const uint8_t *, int, uint32_t *) = isAlpha ? c->alpToYV12 : c->lumToYV12;

    void (*convertRange)(int16_t *, int) = isAlpha ? NULL : c->lumConvertRange;

    const uint8_t *src = src_in[isAlpha ? 3 : 0];



    if (toYV12) {

        toYV12(formatConvBuffer, src, srcW, pal);

        src= formatConvBuffer;

    } else if (c->readLumPlanar && !isAlpha) {

        c->readLumPlanar(formatConvBuffer, src_in, srcW);

        src = formatConvBuffer;

    }



    if (!c->hyscale_fast) {

        c->hyScale(c, dst, dstWidth, src, hLumFilter, hLumFilterPos, hLumFilterSize);

    } else { // fast bilinear upscale / crap downscale

        c->hyscale_fast(c, dst, dstWidth, src, srcW, xInc);

    }



    if (convertRange)

        convertRange(dst, dstWidth);

}
