void rgb32tobgr24(const uint8_t *src, uint8_t *dst, long src_size)

{

	long i;

	long num_pixels = src_size >> 2;

	for(i=0; i<num_pixels; i++)

	{

		#ifdef WORDS_BIGENDIAN

			/* RGB32 (= A,B,G,R) -> BGR24 (= B,G,R) */

			dst[3*i + 0] = src[4*i + 1];

			dst[3*i + 1] = src[4*i + 2];

			dst[3*i + 2] = src[4*i + 3];

		#else

			dst[3*i + 0] = src[4*i + 2];

			dst[3*i + 1] = src[4*i + 1];

			dst[3*i + 2] = src[4*i + 0];

		#endif

	}

}
