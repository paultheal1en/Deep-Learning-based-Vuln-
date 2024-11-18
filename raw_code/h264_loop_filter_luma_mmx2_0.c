static inline void h264_loop_filter_luma_mmx2(uint8_t *pix, int stride, int alpha1, int beta1, int8_t *tc0)

{

    DECLARE_ALIGNED_8(uint64_t, tmp0[2]);



    __asm__ volatile(

        "movq    (%1,%3), %%mm0    \n\t" //p1

        "movq    (%1,%3,2), %%mm1  \n\t" //p0

        "movq    (%2),    %%mm2    \n\t" //q0

        "movq    (%2,%3), %%mm3    \n\t" //q1

        H264_DEBLOCK_MASK(%6, %7)



        "movd      %5,    %%mm4    \n\t"

        "punpcklbw %%mm4, %%mm4    \n\t"

        "punpcklwd %%mm4, %%mm4    \n\t"

        "pcmpeqb   %%mm3, %%mm3    \n\t"

        "movq      %%mm4, %%mm6    \n\t"

        "pcmpgtb   %%mm3, %%mm4    \n\t"

        "movq      %%mm6, 8+%0     \n\t"

        "pand      %%mm4, %%mm7    \n\t"

        "movq      %%mm7, %0       \n\t"



        /* filter p1 */

        "movq     (%1),   %%mm3    \n\t" //p2

        DIFF_GT2_MMX(%%mm1, %%mm3, %%mm5, %%mm6, %%mm4) // |p2-p0|>beta-1

        "pand     %%mm7,  %%mm6    \n\t" // mask & |p2-p0|<beta

        "pand     8+%0,   %%mm7    \n\t" // mask & tc0

        "movq     %%mm7,  %%mm4    \n\t"

        "psubb    %%mm6,  %%mm7    \n\t"

        "pand     %%mm4,  %%mm6    \n\t" // mask & |p2-p0|<beta & tc0

        H264_DEBLOCK_Q1(%%mm0, %%mm3, "(%1)", "(%1,%3)", %%mm6, %%mm4)



        /* filter q1 */

        "movq    (%2,%3,2), %%mm4  \n\t" //q2

        DIFF_GT2_MMX(%%mm2, %%mm4, %%mm5, %%mm6, %%mm3) // |q2-q0|>beta-1

        "pand     %0,     %%mm6    \n\t"

        "movq     8+%0,   %%mm5    \n\t" // can be merged with the and below but is slower then

        "pand     %%mm6,  %%mm5    \n\t"

        "psubb    %%mm6,  %%mm7    \n\t"

        "movq    (%2,%3), %%mm3    \n\t"

        H264_DEBLOCK_Q1(%%mm3, %%mm4, "(%2,%3,2)", "(%2,%3)", %%mm5, %%mm6)



        /* filter p0, q0 */

        H264_DEBLOCK_P0_Q0(%8, unused)

        "movq      %%mm1, (%1,%3,2) \n\t"

        "movq      %%mm2, (%2)      \n\t"



        : "=m"(*tmp0)

        : "r"(pix-3*stride), "r"(pix), "r"((x86_reg)stride),

          "m"(*tmp0/*unused*/), "m"(*(uint32_t*)tc0), "m"(alpha1), "m"(beta1),

          "m"(ff_bone)

    );

}
