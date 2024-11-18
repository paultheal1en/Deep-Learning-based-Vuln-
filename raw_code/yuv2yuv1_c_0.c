static void yuv2yuv1_c(SwsContext *c, const int16_t *lumSrc,

                       const int16_t *chrUSrc, const int16_t *chrVSrc,

                       const int16_t *alpSrc,

                       uint8_t *dest, uint8_t *uDest, uint8_t *vDest,

                       uint8_t *aDest, int dstW, int chrDstW)

{

    int i;

    for (i=0; i<dstW; i++) {

        int val= (lumSrc[i]+64)>>7;

        dest[i]= av_clip_uint8(val);

    }



    if (uDest)

        for (i=0; i<chrDstW; i++) {

            int u=(chrUSrc[i]+64)>>7;

            int v=(chrVSrc[i]+64)>>7;

            uDest[i]= av_clip_uint8(u);

            vDest[i]= av_clip_uint8(v);

        }



    if (CONFIG_SWSCALE_ALPHA && aDest)

        for (i=0; i<dstW; i++) {

            int val= (alpSrc[i]+64)>>7;

            aDest[i]= av_clip_uint8(val);

        }

}
