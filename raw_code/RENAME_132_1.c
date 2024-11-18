static inline void RENAME(rgb24ToY)(uint8_t *dst, uint8_t *src, int width)

{

	int i;

	for(i=0; i<width; i++)

	{

		int r= src[i*3+0];

		int g= src[i*3+1];

		int b= src[i*3+2];



		dst[i]= ((RY*r + GY*g + BY*b + (33<<(RGB2YUV_SHIFT-1)) )>>RGB2YUV_SHIFT);

	}

}
