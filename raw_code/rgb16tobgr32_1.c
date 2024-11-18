void rgb16tobgr32(const uint8_t *src, uint8_t *dst, long src_size)

{

	const uint16_t *end;

	uint8_t *d = (uint8_t *)dst;

	const uint16_t *s = (uint16_t *)src;

	end = s + src_size/2;

	while(s < end)

	{

		register uint16_t bgr;

		bgr = *s++;

		#ifdef WORDS_BIGENDIAN

			*d++ = 0;

			*d++ = (bgr&0x1F)<<3;

			*d++ = (bgr&0x7E0)>>3;

			*d++ = (bgr&0xF800)>>8;

		#else

			*d++ = (bgr&0xF800)>>8;

			*d++ = (bgr&0x7E0)>>3;

			*d++ = (bgr&0x1F)<<3;

			*d++ = 0;

		#endif

	}

}
