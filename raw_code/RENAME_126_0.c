static inline void RENAME(yuv2yuv1)(SwsContext *c, const int16_t *lumSrc,

                                    const int16_t *chrUSrc, const int16_t *chrVSrc,

                                    const int16_t *alpSrc,

                                    uint8_t *dest, uint8_t *uDest, uint8_t *vDest,

                                    uint8_t *aDest, int dstW, int chrDstW)

{

    int p= 4;

    const uint8_t *src[4]= { alpSrc + dstW, lumSrc + dstW, chrUSrc + chrDstW, chrVSrc + chrDstW };

    uint8_t *dst[4]= { aDest, dest, uDest, vDest };

    x86_reg counter[4]= { dstW, dstW, chrDstW, chrDstW };



    while (p--) {

        if (dst[p]) {

            __asm__ volatile(

                "mov %2, %%"REG_a"                    \n\t"

                ".p2align               4             \n\t" /* FIXME Unroll? */

                "1:                                   \n\t"

                "movq  (%0, %%"REG_a", 2), %%mm0      \n\t"

                "movq 8(%0, %%"REG_a", 2), %%mm1      \n\t"

                "psraw                 $7, %%mm0      \n\t"

                "psraw                 $7, %%mm1      \n\t"

                "packuswb           %%mm1, %%mm0      \n\t"

                MOVNTQ(%%mm0, (%1, %%REGa))

                "add                   $8, %%"REG_a"  \n\t"

                "jnc                   1b             \n\t"

                :: "r" (src[p]), "r" (dst[p] + counter[p]),

                   "g" (-counter[p])

                : "%"REG_a

            );

        }

    }

}
