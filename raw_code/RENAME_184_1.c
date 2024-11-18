inline static void RENAME(hcscale)(uint16_t *dst, int dstWidth,

				uint8_t *src1, uint8_t *src2, int srcW, int xInc)

{

#ifdef HAVE_MMX

	// use the new MMX scaler if th mmx2 cant be used (its faster than the x86asm one)

    if(sws_flags != SWS_FAST_BILINEAR || (!canMMX2BeUsed))

#else

    if(sws_flags != SWS_FAST_BILINEAR)

#endif

    {

    	RENAME(hScale)(dst     , dstWidth, src1, srcW, xInc, hChrFilter, hChrFilterPos, hChrFilterSize);

    	RENAME(hScale)(dst+2048, dstWidth, src2, srcW, xInc, hChrFilter, hChrFilterPos, hChrFilterSize);

    }

    else // Fast Bilinear upscale / crap downscale

    {

#ifdef ARCH_X86

#ifdef HAVE_MMX2

	int i;

	if(canMMX2BeUsed)

	{

		asm volatile(

		"pxor %%mm7, %%mm7		\n\t"

		"pxor %%mm2, %%mm2		\n\t" // 2*xalpha

		"movd %5, %%mm6			\n\t" // xInc&0xFFFF

		"punpcklwd %%mm6, %%mm6		\n\t"

		"punpcklwd %%mm6, %%mm6		\n\t"

		"movq %%mm6, %%mm2		\n\t"

		"psllq $16, %%mm2		\n\t"

		"paddw %%mm6, %%mm2		\n\t"

		"psllq $16, %%mm2		\n\t"

		"paddw %%mm6, %%mm2		\n\t"

		"psllq $16, %%mm2		\n\t" //0,t,2t,3t		t=xInc&0xFFFF

		"movq %%mm2, "MANGLE(temp0)"	\n\t"

		"movd %4, %%mm6			\n\t" //(xInc*4)&0xFFFF

		"punpcklwd %%mm6, %%mm6		\n\t"

		"punpcklwd %%mm6, %%mm6		\n\t"

		"xorl %%eax, %%eax		\n\t" // i

		"movl %0, %%esi			\n\t" // src

		"movl %1, %%edi			\n\t" // buf1

		"movl %3, %%edx			\n\t" // (xInc*4)>>16

		"xorl %%ecx, %%ecx		\n\t"

		"xorl %%ebx, %%ebx		\n\t"

		"movw %4, %%bx			\n\t" // (xInc*4)&0xFFFF



#define FUNNYUVCODE \

			PREFETCH" 1024(%%esi)		\n\t"\

			PREFETCH" 1056(%%esi)		\n\t"\

			PREFETCH" 1088(%%esi)		\n\t"\

			"call "MANGLE(funnyUVCode)"	\n\t"\

			"movq "MANGLE(temp0)", %%mm2	\n\t"\

			"xorl %%ecx, %%ecx		\n\t"



FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE



FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE

		"xorl %%eax, %%eax		\n\t" // i

		"movl %6, %%esi			\n\t" // src

		"movl %1, %%edi			\n\t" // buf1

		"addl $4096, %%edi		\n\t"



FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE



FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE

FUNNYUVCODE



		:: "m" (src1), "m" (dst), "m" (dstWidth), "m" ((xInc*4)>>16),

		  "m" ((xInc*4)&0xFFFF), "m" (xInc&0xFFFF), "m" (src2)

		: "%eax", "%ebx", "%ecx", "%edx", "%esi", "%edi"

	);

		for(i=dstWidth-1; (i*xInc)>>16 >=srcW-1; i--)

		{

//			printf("%d %d %d\n", dstWidth, i, srcW);

			dst[i] = src1[srcW-1]*128;

			dst[i+2048] = src2[srcW-1]*128;

		}

	}

	else

	{

#endif

	asm volatile(

		"xorl %%eax, %%eax		\n\t" // i

		"xorl %%ebx, %%ebx		\n\t" // xx

		"xorl %%ecx, %%ecx		\n\t" // 2*xalpha

		".balign 16			\n\t"

		"1:				\n\t"

		"movl %0, %%esi			\n\t"

		"movzbl  (%%esi, %%ebx), %%edi	\n\t" //src[xx]

		"movzbl 1(%%esi, %%ebx), %%esi	\n\t" //src[xx+1]

		"subl %%edi, %%esi		\n\t" //src[xx+1] - src[xx]

		"imull %%ecx, %%esi		\n\t" //(src[xx+1] - src[xx])*2*xalpha

		"shll $16, %%edi		\n\t"

		"addl %%edi, %%esi		\n\t" //src[xx+1]*2*xalpha + src[xx]*(1-2*xalpha)

		"movl %1, %%edi			\n\t"

		"shrl $9, %%esi			\n\t"

		"movw %%si, (%%edi, %%eax, 2)	\n\t"



		"movzbl  (%5, %%ebx), %%edi	\n\t" //src[xx]

		"movzbl 1(%5, %%ebx), %%esi	\n\t" //src[xx+1]

		"subl %%edi, %%esi		\n\t" //src[xx+1] - src[xx]

		"imull %%ecx, %%esi		\n\t" //(src[xx+1] - src[xx])*2*xalpha

		"shll $16, %%edi		\n\t"

		"addl %%edi, %%esi		\n\t" //src[xx+1]*2*xalpha + src[xx]*(1-2*xalpha)

		"movl %1, %%edi			\n\t"

		"shrl $9, %%esi			\n\t"

		"movw %%si, 4096(%%edi, %%eax, 2)\n\t"



		"addw %4, %%cx			\n\t" //2*xalpha += xInc&0xFF

		"adcl %3, %%ebx			\n\t" //xx+= xInc>>8 + carry

		"addl $1, %%eax			\n\t"

		"cmpl %2, %%eax			\n\t"

		" jb 1b				\n\t"



		:: "m" (src1), "m" (dst), "m" (dstWidth), "m" (xInc>>16), "m" (xInc&0xFFFF),

		"r" (src2)

		: "%eax", "%ebx", "%ecx", "%edi", "%esi"

		);

#ifdef HAVE_MMX2

	} //if MMX2 cant be used

#endif

#else

	int i;

	unsigned int xpos=0;

	for(i=0;i<dstWidth;i++)

	{

		register unsigned int xx=xpos>>16;

		register unsigned int xalpha=(xpos&0xFFFF)>>9;

		dst[i]=(src1[xx]*(xalpha^127)+src1[xx+1]*xalpha);

		dst[i+2048]=(src2[xx]*(xalpha^127)+src2[xx+1]*xalpha);

/* slower

	  dst[i]= (src1[xx]<<7) + (src1[xx+1] - src1[xx])*xalpha;

	  dst[i+2048]=(src2[xx]<<7) + (src2[xx+1] - src2[xx])*xalpha;

*/

		xpos+=xInc;

	}

#endif

   }

}
