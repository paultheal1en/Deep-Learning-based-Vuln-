void ff_hcscale_fast_mmxext(SwsContext *c, int16_t *dst1, int16_t *dst2,

                                 int dstWidth, const uint8_t *src1,

                                 const uint8_t *src2, int srcW, int xInc)

{

    int32_t *filterPos = c->hChrFilterPos;

    int16_t *filter    = c->hChrFilter;

    void    *mmxextFilterCode = c->chrMmxextFilterCode;

    int i;

#if ARCH_X86_64

    DECLARE_ALIGNED(8, uint64_t, retsave);

#else

#if defined(PIC)

    DECLARE_ALIGNED(8, uint64_t, ebxsave);

#endif

#endif

    __asm__ volatile(

#if ARCH_X86_64

        "mov          -8(%%rsp), %%"FF_REG_a"    \n\t"

        "mov       %%"FF_REG_a", %7              \n\t"  // retsave

#else

#if defined(PIC)

        "mov       %%"FF_REG_b", %7              \n\t"  // ebxsave

#endif

#endif

        "pxor             %%mm7, %%mm7           \n\t"

        "mov                 %0, %%"FF_REG_c"    \n\t"

        "mov                 %1, %%"FF_REG_D"    \n\t"

        "mov                 %2, %%"FF_REG_d"    \n\t"

        "mov                 %3, %%"FF_REG_b"    \n\t"

        "xor          %%"FF_REG_a", %%"FF_REG_a" \n\t" // i

        PREFETCH"   (%%"FF_REG_c")               \n\t"

        PREFETCH" 32(%%"FF_REG_c")               \n\t"

        PREFETCH" 64(%%"FF_REG_c")               \n\t"



        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        "xor          %%"FF_REG_a", %%"FF_REG_a" \n\t" // i

        "mov                    %5, %%"FF_REG_c" \n\t" // src2

        "mov                    %6, %%"FF_REG_D" \n\t" // dst2

        PREFETCH"   (%%"FF_REG_c")               \n\t"

        PREFETCH" 32(%%"FF_REG_c")               \n\t"

        PREFETCH" 64(%%"FF_REG_c")               \n\t"



        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE



#if ARCH_X86_64

        "mov                    %7, %%"FF_REG_a" \n\t"

        "mov          %%"FF_REG_a", -8(%%rsp)    \n\t"

#else

#if defined(PIC)

        "mov %7, %%"FF_REG_b"    \n\t"

#endif

#endif

        :: "m" (src1), "m" (dst1), "m" (filter), "m" (filterPos),

           "m" (mmxextFilterCode), "m" (src2), "m"(dst2)

#if ARCH_X86_64

          ,"m"(retsave)

#else

#if defined(PIC)

          ,"m" (ebxsave)

#endif

#endif

        : "%"FF_REG_a, "%"FF_REG_c, "%"FF_REG_d, "%"FF_REG_S, "%"FF_REG_D

#if ARCH_X86_64 || !defined(PIC)

         ,"%"FF_REG_b

#endif

    );



    for (i=dstWidth-1; (i*xInc)>>16 >=srcW-1; i--) {

        dst1[i] = src1[srcW-1]*128;

        dst2[i] = src2[srcW-1]*128;

    }

}
