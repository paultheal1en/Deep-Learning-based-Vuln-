static void RENAME(yuv2rgb555_1)(SwsContext *c, const int16_t *buf0,

                                 const int16_t *ubuf[2], const int16_t *bguf[2],

                                 const int16_t *abuf0, uint8_t *dest,

                                 int dstW, int uvalpha, int y)

{

    const int16_t *ubuf0 = ubuf[0], *ubuf1 = ubuf[1];

    const int16_t *buf1= buf0; //FIXME needed for RGB1/BGR1



    if (uvalpha < 2048) { // note this is not correct (shifts chrominance by 0.5 pixels) but it is a bit faster

        __asm__ volatile(

            "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

            "mov        %4, %%"REG_b"               \n\t"

            "push %%"REG_BP"                        \n\t"

            YSCALEYUV2RGB1(%%REGBP, %5)

            "pxor    %%mm7, %%mm7                   \n\t"

            /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

            "paddusb "BLUE_DITHER"(%5), %%mm2      \n\t"

            "paddusb "GREEN_DITHER"(%5), %%mm4      \n\t"

            "paddusb "RED_DITHER"(%5), %%mm5      \n\t"

#endif

            WRITERGB15(%%REGb, 8280(%5), %%REGBP)

            "pop %%"REG_BP"                         \n\t"

            "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"

            :: "c" (buf0), "d" (buf1), "S" (ubuf0), "D" (ubuf1), "m" (dest),

               "a" (&c->redDither)

        );

    } else {

        __asm__ volatile(

            "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

            "mov        %4, %%"REG_b"               \n\t"

            "push %%"REG_BP"                        \n\t"

            YSCALEYUV2RGB1b(%%REGBP, %5)

            "pxor    %%mm7, %%mm7                   \n\t"

            /* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

            "paddusb "BLUE_DITHER"(%5), %%mm2      \n\t"

            "paddusb "GREEN_DITHER"(%5), %%mm4      \n\t"

            "paddusb "RED_DITHER"(%5), %%mm5      \n\t"

#endif

            WRITERGB15(%%REGb, 8280(%5), %%REGBP)

            "pop %%"REG_BP"                         \n\t"

            "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"

            :: "c" (buf0), "d" (buf1), "S" (ubuf0), "D" (ubuf1), "m" (dest),

               "a" (&c->redDither)

        );

    }

}
