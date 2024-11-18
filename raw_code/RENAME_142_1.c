static inline void RENAME(initFilter)(int16_t *filter, int16_t *filterPos, int *filterSize, int xInc,

				      int srcW, int dstW, int filterAlign, int one)

{

	int i;

#ifdef HAVE_MMX

	asm volatile("emms\n\t"::: "memory"); //FIXME this shouldnt be required but it IS (even for non mmx versions) 

#endif



	if(ABS(xInc - 0x10000) <10) // unscaled

	{

		int i;

		*filterSize= (1 +(filterAlign-1)) & (~(filterAlign-1)); // 1 or 4 normaly

		for(i=0; i<dstW*(*filterSize); i++) filter[i]=0;



		for(i=0; i<dstW; i++)

		{

			filter[i*(*filterSize)]=1;

			filterPos[i]=i;

		}



	}

	else if(xInc <= (1<<16) || sws_flags==SWS_FAST_BILINEAR) // upscale

	{

		int i;

		int xDstInSrc;

		if(sws_flags==SWS_BICUBIC) *filterSize= 4;

		else			   *filterSize= 2;

//		printf("%d %d %d\n", filterSize, srcW, dstW);

		*filterSize= (*filterSize +(filterAlign-1)) & (~(filterAlign-1));



		xDstInSrc= xInc - 0x8000;

		for(i=0; i<dstW; i++)

		{

			int xx= (xDstInSrc>>16) - (*filterSize>>1) + 1;

			int j;



			filterPos[i]= xx;

			if(sws_flags == SWS_BICUBIC)

			{

				double d= ABS(((xx+1)<<16) - xDstInSrc)/(double)(1<<16);

//				int coeff;

				int y1,y2,y3,y4;

				double A= -0.75;

					// Equation is from VirtualDub

		y1 = (int)floor(0.5 + (        +     A*d -       2.0*A*d*d +       A*d*d*d) * 16384.0);

		y2 = (int)floor(0.5 + (+ 1.0             -     (A+3.0)*d*d + (A+2.0)*d*d*d) * 16384.0);

		y3 = (int)floor(0.5 + (        -     A*d + (2.0*A+3.0)*d*d - (A+2.0)*d*d*d) * 16384.0);

		y4 = (int)floor(0.5 + (                  +           A*d*d -       A*d*d*d) * 16384.0);



//				printf("%d %d %d \n", coeff, (int)d, xDstInSrc);

				filter[i*(*filterSize) + 0]= y1;

				filter[i*(*filterSize) + 1]= y2;

				filter[i*(*filterSize) + 2]= y3;

				filter[i*(*filterSize) + 3]= y4;

//				printf("%1.3f %d, %d, %d, %d\n",d , y1, y2, y3, y4);

			}

			else

			{

				for(j=0; j<*filterSize; j++)

				{

					double d= ABS((xx<<16) - xDstInSrc)/(double)(1<<16);

					int coeff;

					coeff= (int)(0.5 + (1.0 - d)*(1<<14));

					if(coeff<0) coeff=0;

	//				printf("%d %d %d \n", coeff, (int)d, xDstInSrc);

					filter[i*(*filterSize) + j]= coeff;

					xx++;

				}

			}

			xDstInSrc+= xInc;

		}

	}

	else // downscale

	{

		int xDstInSrc;

		if(sws_flags==SWS_BICUBIC) *filterSize= (int)ceil(1 + 4.0*srcW / (double)dstW);

		else			   *filterSize= (int)ceil(1 + 2.0*srcW / (double)dstW);

//		printf("%d %d %d\n", *filterSize, srcW, dstW);

		*filterSize= (*filterSize +(filterAlign-1)) & (~(filterAlign-1));



		xDstInSrc= xInc - 0x8000;

		for(i=0; i<dstW; i++)

		{

			int xx= (int)((double)xDstInSrc/(double)(1<<16) - *filterSize*0.5 + 0.5);

			int j;



			filterPos[i]= xx;

			for(j=0; j<*filterSize; j++)

			{

				double d= ABS((xx<<16) - xDstInSrc)/(double)xInc;

				int coeff;

				if(sws_flags == SWS_BICUBIC)

				{

					double A= -0.75;

//					d*=2;

					// Equation is from VirtualDub

					if(d<1.0)

						coeff = (int)floor(0.5 + (1.0 - (A+3.0)*d*d

						        + (A+2.0)*d*d*d) * (1<<14));

					else if(d<2.0)

						coeff = (int)floor(0.5 + (-4.0*A + 8.0*A*d

						        - 5.0*A*d*d + A*d*d*d) * (1<<14));

					else

						coeff=0;

				}

				else

				{

					coeff= (int)(0.5 + (1.0 - d)*(1<<14));

					if(coeff<0) coeff=0;

				}

//				if(filterAlign==1) printf("%d %d %d \n", coeff, (int)d, xDstInSrc);

				filter[i*(*filterSize) + j]= coeff;

				xx++;

			}

			xDstInSrc+= xInc;

		}

	}



	//fix borders

	for(i=0; i<dstW; i++)

	{

		int j;

		if(filterPos[i] < 0)

		{

			// Move filter coeffs left to compensate for filterPos

			for(j=1; j<*filterSize; j++)

			{

				int left= MAX(j + filterPos[i], 0);

				filter[i*(*filterSize) + left] += filter[i*(*filterSize) + j];

				filter[i*(*filterSize) + j]=0;

			}

			filterPos[i]= 0;

		}



		if(filterPos[i] + *filterSize > srcW)

		{

			int shift= filterPos[i] + *filterSize - srcW;

			// Move filter coeffs right to compensate for filterPos

			for(j=*filterSize-2; j>=0; j--)

			{

				int right= MIN(j + shift, *filterSize-1);

				filter[i*(*filterSize) +right] += filter[i*(*filterSize) +j];

				filter[i*(*filterSize) +j]=0;

			}

			filterPos[i]= srcW - *filterSize;

		}

	}



	//FIXME try to align filterpos if possible / try to shift filterpos to put zeros at the end

	// and skip these than later



	//Normalize

	for(i=0; i<dstW; i++)

	{

		int j;

		double sum=0;

		double scale= one;

		for(j=0; j<*filterSize; j++)

		{

			sum+= filter[i*(*filterSize) + j];

		}

		scale/= sum;

		for(j=0; j<*filterSize; j++)

		{

			filter[i*(*filterSize) + j]= (int)(filter[i*(*filterSize) + j]*scale);

		}

	}

}
