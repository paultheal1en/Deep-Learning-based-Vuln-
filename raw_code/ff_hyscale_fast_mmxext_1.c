void ff_hyscale_fast_mmxext(SwsContext *c, int16_t *dst,

                                 int dstWidth, const uint8_t *src,

                                 int srcW, int xInc)

{

    int32_t *filterPos = c->hLumFilterPos;

    int16_t *filter    = c->hLumFilter;

    void    *mmxextFilterCode = c->lumMmxextFilterCode;

    int i;

#if ARCH_X86_64

    uint64_t retsave;

#else

#if defined(PIC)

    uint64_t ebxsave;

#endif

#endif



    __asm__ volatile(

#if ARCH_X86_64

        "mov               -8(%%rsp), %%"FF_REG_a"    \n\t"

        "mov            %%"FF_REG_a", %5              \n\t"  // retsave

#else

#if defined(PIC)

        "mov            %%"FF_REG_b", %5              \n\t"  // ebxsave

#endif

#endif

        "pxor                  %%mm7, %%mm7           \n\t"

        "mov                      %0, %%"FF_REG_c"    \n\t"

        "mov                      %1, %%"FF_REG_D"    \n\t"

        "mov                      %2, %%"FF_REG_d"    \n\t"

        "mov                      %3, %%"FF_REG_b"    \n\t"

        "xor            %%"FF_REG_a", %%"FF_REG_a"    \n\t" // i

        PREFETCH"      (%%"FF_REG_c")                 \n\t"

        PREFETCH"    32(%%"FF_REG_c")                 \n\t"

        PREFETCH"    64(%%"FF_REG_c")                 \n\t"



#if ARCH_X86_64

#define CALL_MMXEXT_FILTER_CODE \

        "movl               (%%"FF_REG_b"), %%esi        \n\t"\

        "call                          *%4               \n\t"\

        "movl (%%"FF_REG_b", %%"FF_REG_a"), %%esi        \n\t"\

        "add                  %%"FF_REG_S", %%"FF_REG_c" \n\t"\

        "add                  %%"FF_REG_a", %%"FF_REG_D" \n\t"\

        "xor                  %%"FF_REG_a", %%"FF_REG_a" \n\t"\



#else

#define CALL_MMXEXT_FILTER_CODE \

        "movl               (%%"FF_REG_b"), %%esi        \n\t"\

        "call                          *%4               \n\t"\

        "addl (%%"FF_REG_b", %%"FF_REG_a"), %%"FF_REG_c" \n\t"\

        "add                  %%"FF_REG_a", %%"FF_REG_D" \n\t"\

        "xor                  %%"FF_REG_a", %%"FF_REG_a" \n\t"\



#endif /* ARCH_X86_64 */



        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE

        CALL_MMXEXT_FILTER_CODE



#if ARCH_X86_64

        "mov                      %5, %%"FF_REG_a" \n\t"

        "mov            %%"FF_REG_a", -8(%%rsp)    \n\t"

#else

#if defined(PIC)

        "mov                      %5, %%"FF_REG_b" \n\t"

#endif

#endif

        :: "m" (src), "m" (dst), "m" (filter), "m" (filterPos),

           "m" (mmxextFilterCode)

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



    for (i=dstWidth-1; (i*xInc)>>16 >=srcW-1; i--)

        dst[i] = src[srcW-1]*128;

}
