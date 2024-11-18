static inline void RENAME(palToUV)(uint8_t *dstU, uint8_t *dstV, uint8_t *src1, uint8_t *src2, int width, uint32_t *pal)

{

	int i;

        assert(src1 == src2);

	for(i=0; i<width; i++)

	{

		int p= pal[src1[i]];



		dstU[i]= p>>8;

		dstV[i]= p>>16;

	}

}
