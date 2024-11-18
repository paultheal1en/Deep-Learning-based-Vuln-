static void DEF(avg, pixels8_y2)(uint8_t *block, const uint8_t *pixels, ptrdiff_t line_size, int h)

{

    MOVQ_BFE(mm6);

    __asm__ volatile(

        "lea    (%3, %3), %%"REG_a"     \n\t"

        "movq   (%1), %%mm0             \n\t"

        ".p2align 3                     \n\t"

        "1:                             \n\t"

        "movq   (%1, %3), %%mm1         \n\t"

        "movq   (%1, %%"REG_a"), %%mm2  \n\t"

        PAVGBP(%%mm1, %%mm0, %%mm4,   %%mm2, %%mm1, %%mm5)

        "movq   (%2), %%mm3             \n\t"

        PAVGB_MMX(%%mm3, %%mm4, %%mm0, %%mm6)

        "movq   (%2, %3), %%mm3         \n\t"

        PAVGB_MMX(%%mm3, %%mm5, %%mm1, %%mm6)

        "movq   %%mm0, (%2)             \n\t"

        "movq   %%mm1, (%2, %3)         \n\t"

        "add    %%"REG_a", %1           \n\t"

        "add    %%"REG_a", %2           \n\t"



        "movq   (%1, %3), %%mm1         \n\t"

        "movq   (%1, %%"REG_a"), %%mm0  \n\t"

        PAVGBP(%%mm1, %%mm2, %%mm4,   %%mm0, %%mm1, %%mm5)

        "movq   (%2), %%mm3             \n\t"

        PAVGB_MMX(%%mm3, %%mm4, %%mm2, %%mm6)

        "movq   (%2, %3), %%mm3         \n\t"

        PAVGB_MMX(%%mm3, %%mm5, %%mm1, %%mm6)

        "movq   %%mm2, (%2)             \n\t"

        "movq   %%mm1, (%2, %3)         \n\t"

        "add    %%"REG_a", %1           \n\t"

        "add    %%"REG_a", %2           \n\t"



        "subl   $4, %0                  \n\t"

        "jnz    1b                      \n\t"

        :"+g"(h), "+S"(pixels), "+D"(block)

        :"r"((x86_reg)line_size)

        :REG_a, "memory");

}
