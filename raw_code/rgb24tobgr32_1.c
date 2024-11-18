void rgb24tobgr32(const uint8_t *src, uint8_t *dst, unsigned int src_size)

{

	unsigned i;

	for(i=0; 3*i<src_size; i++)

	{

		dst[4*i + 0] = src[3*i + 2];

		dst[4*i + 1] = src[3*i + 1];

		dst[4*i + 2] = src[3*i + 0];

		dst[4*i + 3] = 0;

	}

}
