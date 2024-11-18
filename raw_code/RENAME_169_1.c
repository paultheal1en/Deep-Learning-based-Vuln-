static inline void RENAME(yuv2yuvX_ar)(SwsContext *c, const int16_t *lumFilter,

                                       const int16_t **lumSrc, int lumFilterSize,

                                       const int16_t *chrFilter, const int16_t **chrUSrc,

                                       const int16_t **chrVSrc,

                                       int chrFilterSize, const int16_t **alpSrc,

                                       uint8_t *dest, uint8_t *uDest, uint8_t *vDest,

                                       uint8_t *aDest, long dstW, long chrDstW)

{

    if (uDest) {

        YSCALEYUV2YV12X_ACCURATE(CHR_MMX_FILTER_OFFSET, uDest, chrDstW, 0)

        YSCALEYUV2YV12X_ACCURATE(CHR_MMX_FILTER_OFFSET, vDest, chrDstW + c->uv_off, c->uv_off)

    }

    if (CONFIG_SWSCALE_ALPHA && aDest) {

        YSCALEYUV2YV12X_ACCURATE(ALP_MMX_FILTER_OFFSET, aDest, dstW, 0)

    }



    YSCALEYUV2YV12X_ACCURATE(LUM_MMX_FILTER_OFFSET, dest, dstW, 0)

}
