static void dct_unquantize_mpeg1_mmx(MpegEncContext *s,

                                     DCTELEM *block, int n, int qscale)

{

    int nCoeffs;

    const UINT16 *quant_matrix;

    

    if(s->alternate_scan) nCoeffs= 64;

    else nCoeffs= nCoeffs= zigzag_end[ s->block_last_index[n] ];



    if (s->mb_intra) {

        int block0;

        if (n < 4) 

            block0 = block[0] * s->y_dc_scale;

        else

            block0 = block[0] * s->c_dc_scale;

        /* XXX: only mpeg1 */

        quant_matrix = s->intra_matrix;

asm volatile(

		"pcmpeqw %%mm7, %%mm7		\n\t"

		"psrlw $15, %%mm7		\n\t"

		"movd %2, %%mm6			\n\t"

		"packssdw %%mm6, %%mm6		\n\t"

		"packssdw %%mm6, %%mm6		\n\t"

                "movl %3, %%eax			\n\t"

		".balign 16\n\t"

		"1:				\n\t"

		"movq (%0, %%eax), %%mm0	\n\t"

		"movq 8(%0, %%eax), %%mm1	\n\t"

		"movq (%1, %%eax), %%mm4	\n\t"

		"movq 8(%1, %%eax), %%mm5	\n\t"

		"pmullw %%mm6, %%mm4		\n\t" // q=qscale*quant_matrix[i]

		"pmullw %%mm6, %%mm5		\n\t" // q=qscale*quant_matrix[i]

		"pxor %%mm2, %%mm2		\n\t"

		"pxor %%mm3, %%mm3		\n\t"

		"pcmpgtw %%mm0, %%mm2		\n\t" // block[i] < 0 ? -1 : 0

		"pcmpgtw %%mm1, %%mm3		\n\t" // block[i] < 0 ? -1 : 0

		"pxor %%mm2, %%mm0		\n\t"

		"pxor %%mm3, %%mm1		\n\t"

		"psubw %%mm2, %%mm0		\n\t" // abs(block[i])

		"psubw %%mm3, %%mm1		\n\t" // abs(block[i])

		"pmullw %%mm4, %%mm0		\n\t" // abs(block[i])*q

		"pmullw %%mm5, %%mm1		\n\t" // abs(block[i])*q

		"pxor %%mm4, %%mm4		\n\t"

		"pxor %%mm5, %%mm5		\n\t" // FIXME slow

		"pcmpeqw (%0, %%eax), %%mm4	\n\t" // block[i] == 0 ? -1 : 0

		"pcmpeqw 8(%0, %%eax), %%mm5	\n\t" // block[i] == 0 ? -1 : 0

		"psraw $3, %%mm0		\n\t"

		"psraw $3, %%mm1		\n\t"

		"psubw %%mm7, %%mm0		\n\t"

		"psubw %%mm7, %%mm1		\n\t"

		"por %%mm7, %%mm0		\n\t"

		"por %%mm7, %%mm1		\n\t"

		"pxor %%mm2, %%mm0		\n\t"

		"pxor %%mm3, %%mm1		\n\t"

		"psubw %%mm2, %%mm0		\n\t"

		"psubw %%mm3, %%mm1		\n\t"

		"pandn %%mm0, %%mm4		\n\t"

		"pandn %%mm1, %%mm5		\n\t"

		"movq %%mm4, (%0, %%eax)	\n\t"

		"movq %%mm5, 8(%0, %%eax)	\n\t"



		"addl $16, %%eax		\n\t"

		"js 1b				\n\t"

		::"r" (block+nCoeffs), "r"(quant_matrix+nCoeffs), "g" (qscale), "g" (-2*nCoeffs)

		: "%eax", "memory"

	);    

        block[0]= block0;



        } else {

        quant_matrix = s->non_intra_matrix;

asm volatile(

		"pcmpeqw %%mm7, %%mm7		\n\t"

		"psrlw $15, %%mm7		\n\t"

		"movd %2, %%mm6			\n\t"

		"packssdw %%mm6, %%mm6		\n\t"

		"packssdw %%mm6, %%mm6		\n\t"

                "movl %3, %%eax			\n\t"

		".balign 16\n\t"

		"1:				\n\t"

		"movq (%0, %%eax), %%mm0	\n\t"

		"movq 8(%0, %%eax), %%mm1	\n\t"

		"movq (%1, %%eax), %%mm4	\n\t"

		"movq 8(%1, %%eax), %%mm5	\n\t"

		"pmullw %%mm6, %%mm4		\n\t" // q=qscale*quant_matrix[i]

		"pmullw %%mm6, %%mm5		\n\t" // q=qscale*quant_matrix[i]

		"pxor %%mm2, %%mm2		\n\t"

		"pxor %%mm3, %%mm3		\n\t"

		"pcmpgtw %%mm0, %%mm2		\n\t" // block[i] < 0 ? -1 : 0

		"pcmpgtw %%mm1, %%mm3		\n\t" // block[i] < 0 ? -1 : 0

		"pxor %%mm2, %%mm0		\n\t"

		"pxor %%mm3, %%mm1		\n\t"

		"psubw %%mm2, %%mm0		\n\t" // abs(block[i])

		"psubw %%mm3, %%mm1		\n\t" // abs(block[i])

		"paddw %%mm0, %%mm0		\n\t" // abs(block[i])*2

		"paddw %%mm1, %%mm1		\n\t" // abs(block[i])*2

		"paddw %%mm7, %%mm0		\n\t" // abs(block[i])*2 + 1

		"paddw %%mm7, %%mm1		\n\t" // abs(block[i])*2 + 1

		"pmullw %%mm4, %%mm0		\n\t" // (abs(block[i])*2 + 1)*q

		"pmullw %%mm5, %%mm1		\n\t" // (abs(block[i])*2 + 1)*q

		"pxor %%mm4, %%mm4		\n\t"

		"pxor %%mm5, %%mm5		\n\t" // FIXME slow

		"pcmpeqw (%0, %%eax), %%mm4	\n\t" // block[i] == 0 ? -1 : 0

		"pcmpeqw 8(%0, %%eax), %%mm5	\n\t" // block[i] == 0 ? -1 : 0

		"psraw $4, %%mm0		\n\t"

		"psraw $4, %%mm1		\n\t"

		"psubw %%mm7, %%mm0		\n\t"

		"psubw %%mm7, %%mm1		\n\t"

		"por %%mm7, %%mm0		\n\t"

		"por %%mm7, %%mm1		\n\t"

		"pxor %%mm2, %%mm0		\n\t"

		"pxor %%mm3, %%mm1		\n\t"

		"psubw %%mm2, %%mm0		\n\t"

		"psubw %%mm3, %%mm1		\n\t"

		"pandn %%mm0, %%mm4		\n\t"

		"pandn %%mm1, %%mm5		\n\t"

		"movq %%mm4, (%0, %%eax)	\n\t"

		"movq %%mm5, 8(%0, %%eax)	\n\t"



		"addl $16, %%eax		\n\t"

		"js 1b				\n\t"

		::"r" (block+nCoeffs), "r"(quant_matrix+nCoeffs), "g" (qscale), "g" (-2*nCoeffs)

		: "%eax", "memory"

	);

    }

}
