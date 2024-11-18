static void add_pixels_clamped_mmx(const DCTELEM *block, UINT8 *pixels, int line_size)

{

    const DCTELEM *p;

    UINT8 *pix;

    int i;



    /* read the pixels */

    p = block;

    pix = pixels;

    MOVQ_ZERO(mm7);

    i = 4;

    while (i) {

	__asm __volatile(

		"movq	%2, %%mm0\n\t"

		"movq	8%2, %%mm1\n\t"

		"movq	16%2, %%mm2\n\t"

		"movq	24%2, %%mm3\n\t"

		"movq	%0, %%mm4\n\t"

		"movq	%1, %%mm6\n\t"

		"movq	%%mm4, %%mm5\n\t"

		"punpcklbw %%mm7, %%mm4\n\t"

		"punpckhbw %%mm7, %%mm5\n\t"

		"paddsw	%%mm4, %%mm0\n\t"

		"paddsw	%%mm5, %%mm1\n\t"

		"movq	%%mm6, %%mm5\n\t"

		"punpcklbw %%mm7, %%mm6\n\t"

		"punpckhbw %%mm7, %%mm5\n\t"

		"paddsw	%%mm6, %%mm2\n\t"

		"paddsw	%%mm5, %%mm3\n\t"

		"packuswb %%mm1, %%mm0\n\t"

		"packuswb %%mm3, %%mm2\n\t"

		"movq	%%mm0, %0\n\t"

		"movq	%%mm2, %1\n\t"

		:"+m"(*pix), "+m"(*(pix+line_size))

		:"m"(*p)

		:"memory");

        pix += line_size*2;

        p += 16;

        i--;

    };

}
