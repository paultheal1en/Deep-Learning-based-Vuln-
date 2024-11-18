static void h263_h_loop_filter_mmx(uint8_t *src, int stride, int qscale)

{

    if (CONFIG_H263_DECODER || CONFIG_H263_ENCODER) {

        const int strength = ff_h263_loop_filter_strength[qscale];

        DECLARE_ALIGNED(8, uint64_t, temp)[4];

        uint8_t *btemp = (uint8_t*)temp;



        src -= 2;



        transpose4x4(btemp,     src,              8, stride);

        transpose4x4(btemp + 4, src + 4 * stride, 8, stride);

        __asm__ volatile (

            H263_LOOP_FILTER // 5 3 4 6



            : "+m"(temp[0]),

              "+m"(temp[1]),

              "+m"(temp[2]),

              "+m"(temp[3])

            : "g"(2 * strength), "m"(ff_pb_FC)

            );



        __asm__ volatile (

            "movq      %%mm5, %%mm1         \n\t"

            "movq      %%mm4, %%mm0         \n\t"

            "punpcklbw %%mm3, %%mm5         \n\t"

            "punpcklbw %%mm6, %%mm4         \n\t"

            "punpckhbw %%mm3, %%mm1         \n\t"

            "punpckhbw %%mm6, %%mm0         \n\t"

            "movq      %%mm5, %%mm3         \n\t"

            "movq      %%mm1, %%mm6         \n\t"

            "punpcklwd %%mm4, %%mm5         \n\t"

            "punpcklwd %%mm0, %%mm1         \n\t"

            "punpckhwd %%mm4, %%mm3         \n\t"

            "punpckhwd %%mm0, %%mm6         \n\t"

            "movd      %%mm5, (%0)          \n\t"

            "punpckhdq %%mm5, %%mm5         \n\t"

            "movd      %%mm5, (%0, %2)      \n\t"

            "movd      %%mm3, (%0, %2, 2)   \n\t"

            "punpckhdq %%mm3, %%mm3         \n\t"

            "movd      %%mm3, (%0, %3)      \n\t"

            "movd      %%mm1, (%1)          \n\t"

            "punpckhdq %%mm1, %%mm1         \n\t"

            "movd      %%mm1, (%1, %2)      \n\t"

            "movd      %%mm6, (%1, %2, 2)   \n\t"

            "punpckhdq %%mm6, %%mm6         \n\t"

            "movd      %%mm6, (%1, %3)      \n\t"

            :: "r"(src),

               "r"(src + 4 * stride),

               "r"((x86_reg)stride),

               "r"((x86_reg)(3 * stride))

            );

    }

}
