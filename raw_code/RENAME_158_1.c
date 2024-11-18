void RENAME(interleaveBytes)(uint8_t *src1, uint8_t *src2, uint8_t *dest,

			    unsigned width, unsigned height, int src1Stride,

			    int src2Stride, int dstStride){

	unsigned h;



	for(h=0; h < height; h++)

	{

		unsigned w;



#ifdef HAVE_MMX

#ifdef HAVE_SSE2

		asm(

			"xor %%"REG_a", %%"REG_a"	\n\t"

			"1:				\n\t"

			PREFETCH" 64(%1, %%"REG_a")	\n\t"

			PREFETCH" 64(%2, %%"REG_a")	\n\t"

			"movdqa (%1, %%"REG_a"), %%xmm0	\n\t"

			"movdqa (%1, %%"REG_a"), %%xmm1	\n\t"

			"movdqa (%2, %%"REG_a"), %%xmm2	\n\t"

			"punpcklbw %%xmm2, %%xmm0	\n\t"

			"punpckhbw %%xmm2, %%xmm1	\n\t"

			"movntdq %%xmm0, (%0, %%"REG_a", 2)\n\t"

			"movntdq %%xmm1, 16(%0, %%"REG_a", 2)\n\t"

			"add $16, %%"REG_a"		\n\t"

			"cmp %3, %%"REG_a"		\n\t"

			" jb 1b				\n\t"

			::"r"(dest), "r"(src1), "r"(src2), "r" ((long)width-15)

			: "memory", "%"REG_a""

		);

#else

		asm(

			"xor %%"REG_a", %%"REG_a"	\n\t"

			"1:				\n\t"

			PREFETCH" 64(%1, %%"REG_a")	\n\t"

			PREFETCH" 64(%2, %%"REG_a")	\n\t"

			"movq (%1, %%"REG_a"), %%mm0	\n\t"

			"movq 8(%1, %%"REG_a"), %%mm2	\n\t"

			"movq %%mm0, %%mm1		\n\t"

			"movq %%mm2, %%mm3		\n\t"

			"movq (%2, %%"REG_a"), %%mm4	\n\t"

			"movq 8(%2, %%"REG_a"), %%mm5	\n\t"

			"punpcklbw %%mm4, %%mm0		\n\t"

			"punpckhbw %%mm4, %%mm1		\n\t"

			"punpcklbw %%mm5, %%mm2		\n\t"

			"punpckhbw %%mm5, %%mm3		\n\t"

			MOVNTQ" %%mm0, (%0, %%"REG_a", 2)\n\t"

			MOVNTQ" %%mm1, 8(%0, %%"REG_a", 2)\n\t"

			MOVNTQ" %%mm2, 16(%0, %%"REG_a", 2)\n\t"

			MOVNTQ" %%mm3, 24(%0, %%"REG_a", 2)\n\t"

			"add $16, %%"REG_a"		\n\t"

			"cmp %3, %%"REG_a"		\n\t"

			" jb 1b				\n\t"

			::"r"(dest), "r"(src1), "r"(src2), "r" ((long)width-15)

			: "memory", "%"REG_a

		);

#endif

		for(w= (width&(~15)); w < width; w++)

		{

			dest[2*w+0] = src1[w];

			dest[2*w+1] = src2[w];

		}

#else

		for(w=0; w < width; w++)

		{

			dest[2*w+0] = src1[w];

			dest[2*w+1] = src2[w];

		}

#endif

		dest += dstStride;

                src1 += src1Stride;

                src2 += src2Stride;

	}

#ifdef HAVE_MMX

	asm(

		EMMS" \n\t"

		SFENCE" \n\t"

		::: "memory"

		);

#endif

}
