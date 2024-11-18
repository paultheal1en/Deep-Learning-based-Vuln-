static inline void hyscale_fast_c(SwsContext *c, int16_t *dst, int dstWidth,

                                  const uint8_t *src, int srcW, int xInc)

{

    int i;

    unsigned int xpos=0;

    for (i=0;i<dstWidth;i++) {

        register unsigned int xx=xpos>>16;

        register unsigned int xalpha=(xpos&0xFFFF)>>9;

        dst[i]= (src[xx]<<7) + (src[xx+1] - src[xx])*xalpha;

        xpos+=xInc;

    }



}