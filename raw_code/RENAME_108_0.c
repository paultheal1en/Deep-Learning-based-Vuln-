static inline void RENAME(yuv2packed2)(SwsContext *c, const uint16_t *buf0, const uint16_t *buf1, const uint16_t *uvbuf0, const uint16_t *uvbuf1,

                          const uint16_t *abuf0, const uint16_t *abuf1, uint8_t *dest, int dstW, int yalpha, int uvalpha, int y)

{

    int  yalpha1=4095- yalpha;

    int uvalpha1=4095-uvalpha;

    int i;



#if COMPILE_TEMPLATE_MMX

    if(!(c->flags & SWS_BITEXACT)) {

        switch(c->dstFormat) {

        //Note 8280 == DSTW_OFFSET but the preprocessor can't handle that there :(

        case PIX_FMT_RGB32:

            if (CONFIG_SWSCALE_ALPHA && c->alpPixBuf) {

#if ARCH_X86_64

                __asm__ volatile(

                    YSCALEYUV2RGB(%%r8, %5)

                    YSCALEYUV2RGB_YA(%%r8, %5, %6, %7)

                    "psraw                  $3, %%mm1       \n\t" /* abuf0[eax] - abuf1[eax] >>7*/

                    "psraw                  $3, %%mm7       \n\t" /* abuf0[eax] - abuf1[eax] >>7*/

                    "packuswb            %%mm7, %%mm1       \n\t"

                    WRITEBGR32(%4, 8280(%5), %%r8, %%mm2, %%mm4, %%mm5, %%mm1, %%mm0, %%mm7, %%mm3, %%mm6)



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "r" (dest),

                    "a" (&c->redDither)

                    ,"r" (abuf0), "r" (abuf1)

                    : "%r8"

                );

#else

                c->u_temp=(intptr_t)abuf0;

                c->v_temp=(intptr_t)abuf1;

                __asm__ volatile(

                    "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                    "mov        %4, %%"REG_b"               \n\t"

                    "push %%"REG_BP"                        \n\t"

                    YSCALEYUV2RGB(%%REGBP, %5)

                    "push                   %0              \n\t"

                    "push                   %1              \n\t"

                    "mov          "U_TEMP"(%5), %0          \n\t"

                    "mov          "V_TEMP"(%5), %1          \n\t"

                    YSCALEYUV2RGB_YA(%%REGBP, %5, %0, %1)

                    "psraw                  $3, %%mm1       \n\t" /* abuf0[eax] - abuf1[eax] >>7*/

                    "psraw                  $3, %%mm7       \n\t" /* abuf0[eax] - abuf1[eax] >>7*/

                    "packuswb            %%mm7, %%mm1       \n\t"

                    "pop                    %1              \n\t"

                    "pop                    %0              \n\t"

                    WRITEBGR32(%%REGb, 8280(%5), %%REGBP, %%mm2, %%mm4, %%mm5, %%mm1, %%mm0, %%mm7, %%mm3, %%mm6)

                    "pop %%"REG_BP"                         \n\t"

                    "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"



                    :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                    "a" (&c->redDither)

                );

#endif

            } else {

                __asm__ volatile(

                    "mov %%"REG_b", "ESP_OFFSET"(%5)        \n\t"

                    "mov        %4, %%"REG_b"               \n\t"

                    "push %%"REG_BP"                        \n\t"

                    YSCALEYUV2RGB(%%REGBP, %5)

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

                YSCALEYUV2RGB(%%REGBP, %5)

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

                YSCALEYUV2RGB(%%REGBP, %5)

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

                YSCALEYUV2RGB(%%REGBP, %5)

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

                "mov %4, %%"REG_b"                        \n\t"

                "push %%"REG_BP"                        \n\t"

                YSCALEYUV2PACKED(%%REGBP, %5)

                WRITEYUY2(%%REGb, 8280(%5), %%REGBP)

                "pop %%"REG_BP"                         \n\t"

                "mov "ESP_OFFSET"(%5), %%"REG_b"        \n\t"

                :: "c" (buf0), "d" (buf1), "S" (uvbuf0), "D" (uvbuf1), "m" (dest),

                "a" (&c->redDither)

            );

            return;

        default: break;

        }

    }

#endif //COMPILE_TEMPLATE_MMX

    YSCALE_YUV_2_ANYRGB_C(YSCALE_YUV_2_RGB2_C, YSCALE_YUV_2_PACKED2_C(void,0), YSCALE_YUV_2_GRAY16_2_C, YSCALE_YUV_2_MONO2_C)

}
