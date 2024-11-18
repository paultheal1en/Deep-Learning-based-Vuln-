static void RENAME(uyvytoyuv420)(uint8_t *ydst, uint8_t *udst, uint8_t *vdst, const uint8_t *src,

                                 int width, int height,

                                 int lumStride, int chromStride, int srcStride)

{

    int y;

    const int chromWidth = FF_CEIL_RSHIFT(width, 1);



    for (y=0; y<height; y++) {

        RENAME(extract_even)(src+1, ydst, width);

        if(y&1) {

            RENAME(extract_even2avg)(src-srcStride, src, udst, vdst, chromWidth);

            udst+= chromStride;

            vdst+= chromStride;

        }



        src += srcStride;

        ydst+= lumStride;

    }

    __asm__(

            EMMS"       \n\t"

            SFENCE"     \n\t"

            ::: "memory"

        );

}
