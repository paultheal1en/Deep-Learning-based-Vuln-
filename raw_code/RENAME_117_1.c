static inline void RENAME(bgr16ToY)(uint8_t *dst, uint8_t *src, int width)

{

	int i;

	for(i=0; i<width; i++)

	{

		int d= ((uint16_t*)src)[i];

		int b= d&0x1F;

		int g= (d>>5)&0x3F;

		int r= (d>>11)&0x1F;



		dst[i]= ((2*RY*r + GY*g + 2*BY*b)>>(RGB2YUV_SHIFT-2)) + 16;

	}

}
