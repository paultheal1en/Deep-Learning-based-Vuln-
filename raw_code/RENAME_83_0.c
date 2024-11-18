static inline void RENAME(yuv2packed1)(SwsContext *c, const uint16_t *buf0, const uint16_t *uvbuf0, const uint16_t *uvbuf1,

                          const uint16_t *abuf0, uint8_t *dest, int dstW, int uvalpha, enum PixelFormat dstFormat, int flags, int y)

{

    const int yalpha1=0;

    int i;



    const uint16_t *buf1= buf0; //FIXME needed for RGB1/BGR1

    const int yalpha= 4096; //FIXME ...



    if (flags&SWS_FULL_CHR_H_INT) {

        c->yuv2packed2(c, buf0, buf0, uvbuf0, uvbuf1, abuf0, abuf0, dest, dstW, 0, uvalpha, y);

        return;

    }



#if COMPILE_TEMPLATE_MMX

    if(!(flags & SWS_BITEXACT)) {

        if (uvalpha < 2048) { // note this is not correct (shifts chrominance by 0.5 pixels) but it is a bit faster

            switch(dstFormat) {

            case PIX_FMT_RGB32:

                if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf) {

                    __asm__ volatile(

                        "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                        "mov        %4, %%"REG_b"               \n\t"

                        "push %%"REG_BP"                        \n\t"

                        YSCALEYUV2RGB1(%%REGBP, %5)

                        YSCALEYUV2RGB1_ALPHA(%%REGBP)

                        WRITEBGR32(%%REGb, 8280(%5), %%REGBP, %%mm2, %%mm4, %%mm5, %%mm7, %%mm0, %%mm1, %%mm3, %%mm6)

                        "pop %%"REG_BP"                         \n\t"

                        "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                        :: "c" (buf0), "d" (abuf0), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                        "a" (&c->redDither)

                    );

                } else {

                    __asm__ volatile(

                        "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                        "mov        %4, %%"REG_b"               \n\t"

                        "push %%"REG_BP"                        \n\t"

                        YSCALEYUV2RGB1(%%REGBP, %5)

                        "pcmpeqd %%mm7, %%mm7                   \n\t"

                        WRITEBGR32(%%REGb, 8280(%5), %%REGBP, %%mm2, %%mm4, %%mm5, %%mm7, %%mm0, %%mm1, %%mm3, %%mm6)

                        "pop %%"REG_BP"                         \n\t"

                        "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                        :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                        "a" (&c->redDither)

                    );

                }

                return;

            case PIX_FMT_BGR24:

                __asm__ volatile(

                    "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                    "mov        %4, %%"REG_b"               \n\t"

                    "push %%"REG_BP"                        \n\t"

                    YSCALEYUV2RGB1(%%REGBP, %5)

                    "pxor    %%mm7, %%mm7                   \n\t"

                    WRITEBGR24(%%REGb, 8280(%5), %%REGBP)

                    "pop %%"REG_BP"                         \n\t"

                    "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            case PIX_FMT_RGB555:

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



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            case PIX_FMT_RGB565:

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



                    WRITERGB16(%%REGb, 8280(%5), %%REGBP)

                    "pop %%"REG_BP"                         \n\t"

                    "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            case PIX_FMT_YUYV422:

                __asm__ volatile(

                    "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                    "mov        %4, %%"REG_b"               \n\t"

                    "push %%"REG_BP"                        \n\t"

                    YSCALEYUV2PACKED1(%%REGBP, %5)

                    WRITEYUY2(%%REGb, 8280(%5), %%REGBP)

                    "pop %%"REG_BP"                         \n\t"

                    "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            }

        } else {

            switch(dstFormat) {

            case PIX_FMT_RGB32:

                if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf) {

                    __asm__ volatile(

                        "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                        "mov        %4, %%"REG_b"               \n\t"

                        "push %%"REG_BP"                        \n\t"

                        YSCALEYUV2RGB1b(%%REGBP, %5)

                        YSCALEYUV2RGB1_ALPHA(%%REGBP)

                        WRITEBGR32(%%REGb, 8280(%5), %%REGBP, %%mm2, %%mm4, %%mm5, %%mm7, %%mm0, %%mm1, %%mm3, %%mm6)

                        "pop %%"REG_BP"                         \n\t"

                        "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                        :: "c" (buf0), "d" (abuf0), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                        "a" (&c->redDither)

                    );

                } else {

                    __asm__ volatile(

                        "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                        "mov        %4, %%"REG_b"               \n\t"

                        "push %%"REG_BP"                        \n\t"

                        YSCALEYUV2RGB1b(%%REGBP, %5)

                        "pcmpeqd %%mm7, %%mm7                   \n\t"

                        WRITEBGR32(%%REGb, 8280(%5), %%REGBP, %%mm2, %%mm4, %%mm5, %%mm7, %%mm0, %%mm1, %%mm3, %%mm6)

                        "pop %%"REG_BP"                         \n\t"

                        "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                        :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                        "a" (&c->redDither)

                    );

                }

                return;

            case PIX_FMT_BGR24:

                __asm__ volatile(

                    "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                    "mov        %4, %%"REG_b"               \n\t"

                    "push %%"REG_BP"                        \n\t"

                    YSCALEYUV2RGB1b(%%REGBP, %5)

                    "pxor    %%mm7, %%mm7                   \n\t"

                    WRITEBGR24(%%REGb, 8280(%5), %%REGBP)

                    "pop %%"REG_BP"                         \n\t"

                    "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            case PIX_FMT_RGB555:

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



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            case PIX_FMT_RGB565:

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



                    WRITERGB16(%%REGb, 8280(%5), %%REGBP)

                    "pop %%"REG_BP"                         \n\t"

                    "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            case PIX_FMT_YUYV422:

                __asm__ volatile(

                    "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                    "mov        %4, %%"REG_b"               \n\t"

                    "push %%"REG_BP"                        \n\t"

                    YSCALEYUV2PACKED1b(%%REGBP, %5)

                    WRITEYUY2(%%REGb, 8280(%5), %%REGBP)

                    "pop %%"REG_BP"                         \n\t"

                    "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

                return;

            }

        }

    }

#endif /* COMPILE_TEMPLATE_MMX */

    if (uvalpha < 2048) {

        YSCALE_YUV_2_ANYRGB_C(YSCALE_YUV_2_RGB1_C, YSCALE_YUV_2_PACKED1_C(void,0), YSCALE_YUV_2_GRAY16_1_C, YSCALE_YUV_2_MONO2_C)

    } else {

        YSCALE_YUV_2_ANYRGB_C(YSCALE_YUV_2_RGB1B_C, YSCALE_YUV_2_PACKED1B_C(void,0), YSCALE_YUV_2_GRAY16_1_C, YSCALE_YUV_2_MONO2_C)

    }

}
