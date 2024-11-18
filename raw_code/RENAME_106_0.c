static inline void RENAME(yuv2nv12X)(SwsContext *c, const int16_t *lumFilter, const int16_t **lumSrc, int lumFilterSize,

                                     const int16_t *chrFilter, const int16_t **chrSrc, int chrFilterSize,

                                     uint8_t *dest, uint8_t *uDest, int dstW, int chrDstW, enum PixelFormat dstFormat)

{

    yuv2nv12XinC(lumFilter, lumSrc, lumFilterSize,

                 chrFilter, chrSrc, chrFilterSize,

                 dest, uDest, dstW, chrDstW, dstFormat);

}
