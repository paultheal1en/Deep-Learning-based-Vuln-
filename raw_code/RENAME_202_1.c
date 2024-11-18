static inline void RENAME(hyscale_fast)(SwsContext *c, int16_t *dst,

                                        int dstWidth, const uint8_t *src, int srcW,

                                        int xInc)

{

#if ARCH_X86

#if COMPILE_TEMPLATE_MMX2

    int32_t *filterPos = c->hLumFilterPos;

    int16_t *filter    = c->hLumFilter;

    int     canMMX2BeUsed  = c->canMMX2BeUsed;

    void    *mmx2FilterCode= c->lumMmx2FilterCode;

    int i;

#if defined(PIC)

    DECLARE_ALIGNED(8, uint64_t, ebxsave);

#endif

    if (canMMX2BeUsed) {

        __asm__ volatile(

#if defined(PIC)

            "mov               %%"REG_b", %5        \n\t"

#endif

            "pxor                  %%mm7, %%mm7     \n\t"

            "mov                      %0, %%"REG_c" \n\t"

            "mov                      %1, %%"REG_D" \n\t"

            "mov                      %2, %%"REG_d" \n\t"

            "mov                      %3, %%"REG_b" \n\t"

            "xor               %%"REG_a", %%"REG_a" \n\t" // i

            PREFETCH"        (%%"REG_c")            \n\t"

            PREFETCH"      32(%%"REG_c")            \n\t"

            PREFETCH"      64(%%"REG_c")            \n\t"



#if ARCH_X86_64



#define CALL_MMX2_FILTER_CODE \

            "movl            (%%"REG_b"), %%esi     \n\t"\

            "call                    *%4            \n\t"\

            "movl (%%"REG_b", %%"REG_a"), %%esi     \n\t"\

            "add               %%"REG_S", %%"REG_c" \n\t"\

            "add               %%"REG_a", %%"REG_D" \n\t"\

            "xor               %%"REG_a", %%"REG_a" \n\t"\



#else



#define CALL_MMX2_FILTER_CODE \

            "movl (%%"REG_b"), %%esi        \n\t"\

            "call         *%4                       \n\t"\

            "addl (%%"REG_b", %%"REG_a"), %%"REG_c" \n\t"\

            "add               %%"REG_a", %%"REG_D" \n\t"\

            "xor               %%"REG_a", %%"REG_a" \n\t"\



#endif /* ARCH_X86_64 */



            CALL_MMX2_FILTER_CODE

            CALL_MMX2_FILTER_CODE

            CALL_MMX2_FILTER_CODE

            CALL_MMX2_FILTER_CODE

            CALL_MMX2_FILTER_CODE

            CALL_MMX2_FILTER_CODE

            CALL_MMX2_FILTER_CODE

            CALL_MMX2_FILTER_CODE



#if defined(PIC)

            "mov                      %5, %%"REG_b" \n\t"

#endif

            :: "m" (src), "m" (dst), "m" (filter), "m" (filterPos),

            "m" (mmx2FilterCode)

#if defined(PIC)

            ,"m" (ebxsave)

#endif

            : "%"REG_a, "%"REG_c, "%"REG_d, "%"REG_S, "%"REG_D

#if !defined(PIC)

            ,"%"REG_b

#endif

        );

        for (i=dstWidth-1; (i*xInc)>>16 >=srcW-1; i--) dst[i] = src[srcW-1]*128;

    } else {

#endif /* COMPILE_TEMPLATE_MMX2 */

    x86_reg dstWidth_reg = dstWidth;

    x86_reg xInc_shr16 = xInc >> 16;

    uint16_t xInc_mask = xInc & 0xffff;

    //NO MMX just normal asm ...

    __asm__ volatile(

        "xor %%"REG_a", %%"REG_a"            \n\t" // i

        "xor %%"REG_d", %%"REG_d"            \n\t" // xx

        "xorl    %%ecx, %%ecx                \n\t" // xalpha

        ASMALIGN(4)

        "1:                                  \n\t"

        "movzbl    (%0, %%"REG_d"), %%edi    \n\t" //src[xx]

        "movzbl   1(%0, %%"REG_d"), %%esi    \n\t" //src[xx+1]

        FAST_BILINEAR_X86

        "movw     %%si, (%%"REG_D", %%"REG_a", 2)   \n\t"

        "addw       %4, %%cx                 \n\t" //xalpha += xInc&0xFFFF

        "adc        %3, %%"REG_d"            \n\t" //xx+= xInc>>16 + carry



        "movzbl    (%0, %%"REG_d"), %%edi    \n\t" //src[xx]

        "movzbl   1(%0, %%"REG_d"), %%esi    \n\t" //src[xx+1]

        FAST_BILINEAR_X86

        "movw     %%si, 2(%%"REG_D", %%"REG_a", 2)  \n\t"

        "addw       %4, %%cx                 \n\t" //xalpha += xInc&0xFFFF

        "adc        %3, %%"REG_d"            \n\t" //xx+= xInc>>16 + carry





        "add        $2, %%"REG_a"            \n\t"

        "cmp        %2, %%"REG_a"            \n\t"

        " jb        1b                       \n\t"





        :: "r" (src), "m" (dst), "m" (dstWidth_reg), "m" (xInc_shr16), "m" (xInc_mask)

        : "%"REG_a, "%"REG_d, "%ecx", "%"REG_D, "%esi"

    );

#if COMPILE_TEMPLATE_MMX2

    } //if MMX2 can't be used

#endif

#else

    int i;

    unsigned int xpos=0;

    for (i=0;i<dstWidth;i++) {

        register unsigned int xx=xpos>>16;

        register unsigned int xalpha=(xpos&0xFFFF)>>9;

        dst[i]= (src[xx]<<7) + (src[xx+1] - src[xx])*xalpha;

        xpos+=xInc;

    }

#endif /* ARCH_X86 */

}
