static inline void RENAME(yuv2yuvX)(SwsContext *c, const int16_t *lumFilter, const int16_t **lumSrc, int lumFilterSize,

                                    const int16_t *chrFilter, const int16_t **chrSrc, int chrFilterSize, const int16_t **alpSrc,

                                    uint8_t *dest, uint8_t *uDest, uint8_t *vDest, uint8_t *aDest, int dstW, int chrDstW)

{

#if COMPILE_TEMPLATE_MMX

    if(!(c->flags & SWS_BITEXACT)) {

        if (c->flags & SWS_ACCURATE_RND) {

            if (uDest) {

                YSCALEYUV2YV12X_ACCURATE(   "0", CHR_MMX_FILTER_OFFSET, uDest, chrDstW)

                YSCALEYUV2YV12X_ACCURATE(AV_STRINGIFY(VOF), CHR_MMX_FILTER_OFFSET, vDest, chrDstW)

            }

            if (CONFIG_SWSCALE_ALPHA && aDest) {

                YSCALEYUV2YV12X_ACCURATE(   "0", ALP_MMX_FILTER_OFFSET, aDest, dstW)

            }



            YSCALEYUV2YV12X_ACCURATE("0", LUM_MMX_FILTER_OFFSET, dest, dstW)

        } else {

            if (uDest) {

                YSCALEYUV2YV12X(   "0", CHR_MMX_FILTER_OFFSET, uDest, chrDstW)

                YSCALEYUV2YV12X(AV_STRINGIFY(VOF), CHR_MMX_FILTER_OFFSET, vDest, chrDstW)

            }

            if (CONFIG_SWSCALE_ALPHA && aDest) {

                YSCALEYUV2YV12X(   "0", ALP_MMX_FILTER_OFFSET, aDest, dstW)

            }



            YSCALEYUV2YV12X("0", LUM_MMX_FILTER_OFFSET, dest, dstW)

        }

        return;

    }

#endif

#if COMPILE_TEMPLATE_ALTIVEC

    yuv2yuvX_altivec_real(lumFilter, lumSrc, lumFilterSize,

                          chrFilter, chrSrc, chrFilterSize,

                          dest, uDest, vDest, dstW, chrDstW);

#else //COMPILE_TEMPLATE_ALTIVEC

    yuv2yuvXinC(lumFilter, lumSrc, lumFilterSize,

                chrFilter, chrSrc, chrFilterSize,

                alpSrc, dest, uDest, vDest, aDest, dstW, chrDstW);

#endif //!COMPILE_TEMPLATE_ALTIVEC

}
