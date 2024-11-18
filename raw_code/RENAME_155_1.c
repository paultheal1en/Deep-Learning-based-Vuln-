static inline void RENAME(uyvytoyv12)(const uint8_t *src, uint8_t *ydst, uint8_t *udst, uint8_t *vdst,

	unsigned int width, unsigned int height,

	int lumStride, int chromStride, int srcStride)

{

	unsigned y;

	const unsigned chromWidth= width>>1;

	for(y=0; y<height; y+=2)

	{

#ifdef HAVE_MMX

		asm volatile(

			"xorl %%eax, %%eax		\n\t"

			"pcmpeqw %%mm7, %%mm7		\n\t"

			"psrlw $8, %%mm7		\n\t" // FF,00,FF,00...

			".balign 16			\n\t"

			"1:				\n\t"

			PREFETCH" 64(%0, %%eax, 4)	\n\t"

			"movq (%0, %%eax, 4), %%mm0	\n\t" // UYVY UYVY(0)

			"movq 8(%0, %%eax, 4), %%mm1	\n\t" // UYVY UYVY(4)

			"movq %%mm0, %%mm2		\n\t" // UYVY UYVY(0)

			"movq %%mm1, %%mm3		\n\t" // UYVY UYVY(4)

			"pand %%mm7, %%mm0		\n\t" // U0V0 U0V0(0)

			"pand %%mm7, %%mm1		\n\t" // U0V0 U0V0(4)

			"psrlw $8, %%mm2		\n\t" // Y0Y0 Y0Y0(0)

			"psrlw $8, %%mm3		\n\t" // Y0Y0 Y0Y0(4)

			"packuswb %%mm1, %%mm0		\n\t" // UVUV UVUV(0)

			"packuswb %%mm3, %%mm2		\n\t" // YYYY YYYY(0)



			MOVNTQ" %%mm2, (%1, %%eax, 2)	\n\t"



			"movq 16(%0, %%eax, 4), %%mm1	\n\t" // UYVY UYVY(8)

			"movq 24(%0, %%eax, 4), %%mm2	\n\t" // UYVY UYVY(12)

			"movq %%mm1, %%mm3		\n\t" // UYVY UYVY(8)

			"movq %%mm2, %%mm4		\n\t" // UYVY UYVY(12)

			"pand %%mm7, %%mm1		\n\t" // U0V0 U0V0(8)

			"pand %%mm7, %%mm2		\n\t" // U0V0 U0V0(12)

			"psrlw $8, %%mm3		\n\t" // Y0Y0 Y0Y0(8)

			"psrlw $8, %%mm4		\n\t" // Y0Y0 Y0Y0(12)

			"packuswb %%mm2, %%mm1		\n\t" // UVUV UVUV(8)

			"packuswb %%mm4, %%mm3		\n\t" // YYYY YYYY(8)



			MOVNTQ" %%mm3, 8(%1, %%eax, 2)	\n\t"



			"movq %%mm0, %%mm2		\n\t" // UVUV UVUV(0)

			"movq %%mm1, %%mm3		\n\t" // UVUV UVUV(8)

			"psrlw $8, %%mm0		\n\t" // V0V0 V0V0(0)

			"psrlw $8, %%mm1		\n\t" // V0V0 V0V0(8)

			"pand %%mm7, %%mm2		\n\t" // U0U0 U0U0(0)

			"pand %%mm7, %%mm3		\n\t" // U0U0 U0U0(8)

			"packuswb %%mm1, %%mm0		\n\t" // VVVV VVVV(0)

			"packuswb %%mm3, %%mm2		\n\t" // UUUU UUUU(0)



			MOVNTQ" %%mm0, (%3, %%eax)	\n\t"

			MOVNTQ" %%mm2, (%2, %%eax)	\n\t"



			"addl $8, %%eax			\n\t"

			"cmpl %4, %%eax			\n\t"

			" jb 1b				\n\t"

			::"r"(src), "r"(ydst), "r"(udst), "r"(vdst), "g" (chromWidth)

			: "memory", "%eax"

		);



		ydst += lumStride;

		src  += srcStride;



		asm volatile(

			"xorl %%eax, %%eax		\n\t"

			".balign 16			\n\t"

			"1:				\n\t"

			PREFETCH" 64(%0, %%eax, 4)	\n\t"

			"movq (%0, %%eax, 4), %%mm0	\n\t" // YUYV YUYV(0)

			"movq 8(%0, %%eax, 4), %%mm1	\n\t" // YUYV YUYV(4)

			"movq 16(%0, %%eax, 4), %%mm2	\n\t" // YUYV YUYV(8)

			"movq 24(%0, %%eax, 4), %%mm3	\n\t" // YUYV YUYV(12)

			"psrlw $8, %%mm0		\n\t" // Y0Y0 Y0Y0(0)

			"psrlw $8, %%mm1		\n\t" // Y0Y0 Y0Y0(4)

			"psrlw $8, %%mm2		\n\t" // Y0Y0 Y0Y0(8)

			"psrlw $8, %%mm3		\n\t" // Y0Y0 Y0Y0(12)

			"packuswb %%mm1, %%mm0		\n\t" // YYYY YYYY(0)

			"packuswb %%mm3, %%mm2		\n\t" // YYYY YYYY(8)



			MOVNTQ" %%mm0, (%1, %%eax, 2)	\n\t"

			MOVNTQ" %%mm2, 8(%1, %%eax, 2)	\n\t"



			"addl $8, %%eax			\n\t"

			"cmpl %4, %%eax			\n\t"

			" jb 1b				\n\t"



			::"r"(src), "r"(ydst), "r"(udst), "r"(vdst), "g" (chromWidth)

			: "memory", "%eax"

		);

#else

		unsigned i;

		for(i=0; i<chromWidth; i++)

		{

			udst[i] 	= src[4*i+0];

			ydst[2*i+0] 	= src[4*i+1];

			vdst[i] 	= src[4*i+2];

			ydst[2*i+1] 	= src[4*i+3];

		}

		ydst += lumStride;

		src  += srcStride;



		for(i=0; i<chromWidth; i++)

		{

			ydst[2*i+0] 	= src[4*i+1];

			ydst[2*i+1] 	= src[4*i+3];

		}

#endif

		udst += chromStride;

		vdst += chromStride;

		ydst += lumStride;

		src  += srcStride;

	}

#ifdef HAVE_MMX

asm volatile(   EMMS" \n\t"

        	SFENCE" \n\t"

        	:::"memory");

#endif

}
