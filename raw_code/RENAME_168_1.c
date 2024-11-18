static inline void RENAME(rgb15to32)(const uint8_t *src, uint8_t *dst, long src_size)

{

	const uint16_t *end;

#ifdef HAVE_MMX

	const uint16_t *mm_end;

#endif

	uint8_t *d = (uint8_t *)dst;

	const uint16_t *s = (const uint16_t *)src;

	end = s + src_size/2;

#ifdef HAVE_MMX

	__asm __volatile(PREFETCH"	%0"::"m"(*s):"memory");

	__asm __volatile("pxor	%%mm7,%%mm7\n\t":::"memory");

	mm_end = end - 3;

	while(s < mm_end)

	{

	    __asm __volatile(

		PREFETCH" 32%1\n\t"

		"movq	%1, %%mm0\n\t"

		"movq	%1, %%mm1\n\t"

		"movq	%1, %%mm2\n\t"

		"pand	%2, %%mm0\n\t"

		"pand	%3, %%mm1\n\t"

		"pand	%4, %%mm2\n\t"

		"psllq	$3, %%mm0\n\t"

		"psrlq	$2, %%mm1\n\t"

		"psrlq	$7, %%mm2\n\t"

		"movq	%%mm0, %%mm3\n\t"

		"movq	%%mm1, %%mm4\n\t"

		"movq	%%mm2, %%mm5\n\t"

		"punpcklwd %%mm7, %%mm0\n\t"

		"punpcklwd %%mm7, %%mm1\n\t"

		"punpcklwd %%mm7, %%mm2\n\t"

		"punpckhwd %%mm7, %%mm3\n\t"

		"punpckhwd %%mm7, %%mm4\n\t"

		"punpckhwd %%mm7, %%mm5\n\t"

		"psllq	$8, %%mm1\n\t"

		"psllq	$16, %%mm2\n\t"

		"por	%%mm1, %%mm0\n\t"

		"por	%%mm2, %%mm0\n\t"

		"psllq	$8, %%mm4\n\t"

		"psllq	$16, %%mm5\n\t"

		"por	%%mm4, %%mm3\n\t"

		"por	%%mm5, %%mm3\n\t"

		MOVNTQ"	%%mm0, %0\n\t"

		MOVNTQ"	%%mm3, 8%0\n\t"

		:"=m"(*d)

		:"m"(*s),"m"(mask15b),"m"(mask15g),"m"(mask15r)

		:"memory");

		d += 16;

		s += 4;

	}

	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");

#endif

	while(s < end)

	{

#if 0 //slightly slower on athlon

		int bgr= *s++;

		*((uint32_t*)d)++ = ((bgr&0x1F)<<3) + ((bgr&0x3E0)<<6) + ((bgr&0x7C00)<<9);

#else

		register uint16_t bgr;

		bgr = *s++;

#ifdef WORDS_BIGENDIAN

		*d++ = 0;

		*d++ = (bgr&0x7C00)>>7;

		*d++ = (bgr&0x3E0)>>2;

		*d++ = (bgr&0x1F)<<3;

#else

		*d++ = (bgr&0x1F)<<3;

		*d++ = (bgr&0x3E0)>>2;

		*d++ = (bgr&0x7C00)>>7;

		*d++ = 0;

#endif



#endif

	}

}
