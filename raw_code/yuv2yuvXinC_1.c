static inline void yuv2yuvXinC(int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				    int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

				    uint8_t *dest, uint8_t *uDest, uint8_t *vDest, int dstW, int chrDstW)

{

	//FIXME Optimize (just quickly writen not opti..)

	int i;

	for(i=0; i<dstW; i++)

	{

		int val=1<<18;

		int j;

		for(j=0; j<lumFilterSize; j++)

			val += lumSrc[j][i] * lumFilter[j];



		dest[i]= av_clip_uint8(val>>19);

	}



	if(uDest != NULL)

		for(i=0; i<chrDstW; i++)

		{

			int u=1<<18;

			int v=1<<18;

			int j;

			for(j=0; j<chrFilterSize; j++)

			{

				u += chrSrc[j][i] * chrFilter[j];

				v += chrSrc[j][i + 2048] * chrFilter[j];

			}



			uDest[i]= av_clip_uint8(u>>19);

			vDest[i]= av_clip_uint8(v>>19);

		}

}
