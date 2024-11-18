static inline void RENAME(palToY)(uint8_t *dst, uint8_t *src, int width, uint32_t *pal)

{

	int i;

	for(i=0; i<width; i++)

	{

		int d= src[i];



		dst[i]= pal[d] & 0xFF;

	}

}
