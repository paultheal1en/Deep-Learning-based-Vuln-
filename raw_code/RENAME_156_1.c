static inline void RENAME(yuv2yuv1)(int16_t *lumSrc, int16_t *chrSrc,

				    uint8_t *dest, uint8_t *uDest, uint8_t *vDest, long dstW, long chrDstW)

{

#ifdef HAVE_MMX

	if(uDest != NULL)

	{

		asm volatile(

				YSCALEYUV2YV121

				:: "r" (chrSrc + chrDstW), "r" (uDest + chrDstW),

				"g" (-chrDstW)

				: "%"REG_a

			);



		asm volatile(

				YSCALEYUV2YV121

				:: "r" (chrSrc + 2048 + chrDstW), "r" (vDest + chrDstW),

				"g" (-chrDstW)

				: "%"REG_a

			);

	}



	asm volatile(

		YSCALEYUV2YV121

		:: "r" (lumSrc + dstW), "r" (dest + dstW),

		"g" (-dstW)

		: "%"REG_a

	);

#else

	int i;

	for(i=0; i<dstW; i++)

	{

		int val= lumSrc[i]>>7;



		if(val&256){

			if(val<0) val=0;

			else      val=255;

		}



		dest[i]= val;

	}



	if(uDest != NULL)

		for(i=0; i<chrDstW; i++)

		{

			int u=chrSrc[i]>>7;

			int v=chrSrc[i + 2048]>>7;



			if((u|v)&256){

				if(u<0)         u=0;

				else if (u>255) u=255;

				if(v<0)         v=0;

				else if (v>255) v=255;

			}



			uDest[i]= u;

			vDest[i]= v;

		}

#endif

}
