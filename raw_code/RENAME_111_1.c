static inline void RENAME(bgr15ToY)(uint8_t *dst, uint8_t *src, int width)

{

	int i;

	for(i=0; i<width; i++)

	{

		int d= ((uint16_t*)src)[i];

		int b= d&0x1F;

		int g= (d>>5)&0x1F;

		int r= (d>>10)&0x1F;



		dst[i]= ((RY*r + GY*g + BY*b)>>(RGB2YUV_SHIFT-3)) + 16;

	}

}
