inline static void RENAME(hcscale)(SwsContext *c, uint16_t *dst, long dstWidth, const uint8_t *src1, const uint8_t *src2,

                                   int srcW, int xInc, int flags, const int16_t *hChrFilter,

                                   const int16_t *hChrFilterPos, int hChrFilterSize,

                                   enum PixelFormat srcFormat, uint8_t *formatConvBuffer,

                                   uint32_t *pal)

{

    int32_t av_unused *mmx2FilterPos = c->chrMmx2FilterPos;

    int16_t av_unused *mmx2Filter    = c->chrMmx2Filter;

    int     av_unused canMMX2BeUsed  = c->canMMX2BeUsed;

    void    av_unused *mmx2FilterCode= c->chrMmx2FilterCode;



    if (isGray(srcFormat) || srcFormat==PIX_FMT_MONOBLACK || srcFormat==PIX_FMT_MONOWHITE)

        return;



    src1 += c->chrSrcOffset;

    src2 += c->chrSrcOffset;



    if (c->hcscale_internal) {

        c->hcscale_internal(formatConvBuffer, formatConvBuffer+VOFW, src1, src2, srcW, pal);

        src1= formatConvBuffer;

        src2= formatConvBuffer+VOFW;

    }



    if (!c->hcscale_fast)

    {

        c->hScale(dst     , dstWidth, src1, srcW, xInc, hChrFilter, hChrFilterPos, hChrFilterSize);

        c->hScale(dst+VOFW, dstWidth, src2, srcW, xInc, hChrFilter, hChrFilterPos, hChrFilterSize);

    } else { // fast bilinear upscale / crap downscale

#if ARCH_X86 && CONFIG_GPL

#if COMPILE_TEMPLATE_MMX2

        int i;

#if defined(PIC)

        DECLARE_ALIGNED(8, uint64_t, ebxsave);

#endif

        if (canMMX2BeUsed) {

            __asm__ volatile(

#if defined(PIC)

                "mov          %%"REG_b", %6         \n\t"

#endif

                "pxor             %%mm7, %%mm7      \n\t"

                "mov                 %0, %%"REG_c"  \n\t"

                "mov                 %1, %%"REG_D"  \n\t"

                "mov                 %2, %%"REG_d"  \n\t"

                "mov                 %3, %%"REG_b"  \n\t"

                "xor          %%"REG_a", %%"REG_a"  \n\t" // i

                PREFETCH"   (%%"REG_c")             \n\t"

                PREFETCH" 32(%%"REG_c")             \n\t"

                PREFETCH" 64(%%"REG_c")             \n\t"



                CALL_MMX2_FILTER_CODE

                CALL_MMX2_FILTER_CODE

                CALL_MMX2_FILTER_CODE

                CALL_MMX2_FILTER_CODE

                "xor          %%"REG_a", %%"REG_a"  \n\t" // i

                "mov                 %5, %%"REG_c"  \n\t" // src

                "mov                 %1, %%"REG_D"  \n\t" // buf1

                "add              $"AV_STRINGIFY(VOF)", %%"REG_D"  \n\t"

                PREFETCH"   (%%"REG_c")             \n\t"

                PREFETCH" 32(%%"REG_c")             \n\t"

                PREFETCH" 64(%%"REG_c")             \n\t"



                CALL_MMX2_FILTER_CODE

                CALL_MMX2_FILTER_CODE

                CALL_MMX2_FILTER_CODE

                CALL_MMX2_FILTER_CODE



#if defined(PIC)

                "mov %6, %%"REG_b"    \n\t"

#endif

                :: "m" (src1), "m" (dst), "m" (mmx2Filter), "m" (mmx2FilterPos),

                "m" (mmx2FilterCode), "m" (src2)

#if defined(PIC)

                ,"m" (ebxsave)

#endif

                : "%"REG_a, "%"REG_c, "%"REG_d, "%"REG_S, "%"REG_D

#if !defined(PIC)

                ,"%"REG_b

#endif

            );

            for (i=dstWidth-1; (i*xInc)>>16 >=srcW-1; i--) {

                //printf("%d %d %d\n", dstWidth, i, srcW);

                dst[i] = src1[srcW-1]*128;

                dst[i+VOFW] = src2[srcW-1]*128;

            }

        } else {

#endif /* COMPILE_TEMPLATE_MMX2 */

            x86_reg xInc_shr16 = (x86_reg) (xInc >> 16);

            uint16_t xInc_mask = xInc & 0xffff;

            __asm__ volatile(

                "xor %%"REG_a", %%"REG_a"               \n\t" // i

                "xor %%"REG_d", %%"REG_d"               \n\t" // xx

                "xorl    %%ecx, %%ecx                   \n\t" // xalpha

                ASMALIGN(4)

                "1:                                     \n\t"

                "mov        %0, %%"REG_S"               \n\t"

                "movzbl  (%%"REG_S", %%"REG_d"), %%edi  \n\t" //src[xx]

                "movzbl 1(%%"REG_S", %%"REG_d"), %%esi  \n\t" //src[xx+1]

                FAST_BILINEAR_X86

                "movw     %%si, (%%"REG_D", %%"REG_a", 2)   \n\t"



                "movzbl    (%5, %%"REG_d"), %%edi       \n\t" //src[xx]

                "movzbl   1(%5, %%"REG_d"), %%esi       \n\t" //src[xx+1]

                FAST_BILINEAR_X86

                "movw     %%si, "AV_STRINGIFY(VOF)"(%%"REG_D", %%"REG_a", 2)   \n\t"



                "addw       %4, %%cx                    \n\t" //xalpha += xInc&0xFFFF

                "adc        %3, %%"REG_d"               \n\t" //xx+= xInc>>16 + carry

                "add        $1, %%"REG_a"               \n\t"

                "cmp        %2, %%"REG_a"               \n\t"

                " jb        1b                          \n\t"



/* GCC 3.3 makes MPlayer crash on IA-32 machines when using "g" operand here,

   which is needed to support GCC 4.0. */

#if ARCH_X86_64 && AV_GCC_VERSION_AT_LEAST(3,4)

                :: "m" (src1), "m" (dst), "g" (dstWidth), "m" (xInc_shr16), "m" (xInc_mask),

#else

                :: "m" (src1), "m" (dst), "m" (dstWidth), "m" (xInc_shr16), "m" (xInc_mask),

#endif

                "r" (src2)

                : "%"REG_a, "%"REG_d, "%ecx", "%"REG_D, "%esi"

            );

#if COMPILE_TEMPLATE_MMX2

        } //if MMX2 can't be used

#endif

#else

        c->hcscale_fast(c, dst, dstWidth, src1, src2, srcW, xInc);

#endif /* ARCH_X86 */

    }



    if (c->chrConvertRange)

        c->chrConvertRange(dst, dstWidth);

}
