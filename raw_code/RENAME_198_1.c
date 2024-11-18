static void RENAME(SwScale_YV12slice)(unsigned char* srcptr[],int stride[], int srcSliceY ,

			     int srcSliceH, uint8_t* dstptr[], int dststride, int dstbpp,

			     int srcW, int srcH, int dstW, int dstH){





unsigned int lumXInc= (srcW << 16) / dstW;

unsigned int lumYInc= (srcH << 16) / dstH;

unsigned int chrXInc;

unsigned int chrYInc;



static int dstY;



// used to detect a size change

static int oldDstW= -1;

static int oldSrcW= -1;

static int oldDstH= -1;

static int oldSrcH= -1;

static int oldFlags=-1;



static int lastInLumBuf;

static int lastInChrBuf;



int chrDstW, chrDstH;



static int lumBufIndex=0;

static int chrBufIndex=0;



static int firstTime=1;



int widthAlign= dstbpp==12 ? 16 : 8;

if(((dstW + widthAlign-1)&(~(widthAlign-1))) > dststride)

{

	dstW&= ~(widthAlign-1);

	if(firstTime)

		fprintf(stderr, "SwScaler: Warning: dstStride is not a multiple of %d!\n"

				"SwScaler: ->lowering width to compensate, new width=%d\n"

				"SwScaler: ->cannot do aligned memory acesses anymore\n",

				widthAlign, dstW);

}



//printf("%d %d %d %d\n", srcW, srcH, dstW, dstH);

//printf("%d %d %d %d\n", lumXInc, lumYInc, srcSliceY, srcSliceH);



#ifdef HAVE_MMX2

canMMX2BeUsed= (lumXInc <= 0x10000 && (dstW&31)==0 && (srcW&15)==0) ? 1 : 0;

if(!canMMX2BeUsed && lumXInc <= 0x10000 && (srcW&15)==0 && sws_flags==SWS_FAST_BILINEAR)

{

	if(firstTime) //FIXME only if verbose ?

		fprintf(stderr, "SwScaler: output Width is not a multiple of 32 -> no MMX2 scaler\n");

}

#endif



if(firstTime)

{

#if defined (DITHER1XBPP) && defined (HAVE_MMX)

	char *dither= " dithered";

#else

	char *dither= "";

#endif

	if(sws_flags==SWS_FAST_BILINEAR)

		fprintf(stderr, "SwScaler: FAST_BILINEAR scaler ");

	else if(sws_flags==SWS_BILINEAR)

		fprintf(stderr, "SwScaler: BILINEAR scaler ");

	else if(sws_flags==SWS_BICUBIC)

		fprintf(stderr, "SwScaler: BICUBIC scaler ");

	else

		fprintf(stderr, "SwScaler: ehh flags invalid?! ");



	if(dstbpp==15)

		fprintf(stderr, "with%s BGR15 output ", dither);

	else if(dstbpp==16)

		fprintf(stderr, "with%s BGR16 output ", dither);

	else if(dstbpp==24)

		fprintf(stderr, "with BGR24 output ");

	else if(dstbpp==32)

		fprintf(stderr, "with BGR32 output ");

	else if(dstbpp==12)

		fprintf(stderr, "with YV12 output ");

	else

		fprintf(stderr, "without output ");



#ifdef HAVE_MMX2

		fprintf(stderr, "using MMX2\n");

#elif defined (HAVE_3DNOW)

		fprintf(stderr, "using 3DNOW\n");

#elif defined (HAVE_MMX)

		fprintf(stderr, "using MMX\n");

#elif defined (ARCH_X86)

		fprintf(stderr, "using X86 ASM2\n");

#else

		fprintf(stderr, "using C\n");

#endif

}





// match pixel 0 of the src to pixel 0 of dst and match pixel n-2 of src to pixel n-2 of dst

// n-2 is the last chrominance sample available

// this is not perfect, but noone shuld notice the difference, the more correct variant

// would be like the vertical one, but that would require some special code for the

// first and last pixel

if(sws_flags==SWS_FAST_BILINEAR)

{

	if(canMMX2BeUsed) 	lumXInc+= 20;

	else			lumXInc = ((srcW-2)<<16)/(dstW-2) - 20;

}



if(fullUVIpol && !(dstbpp==12)) 	chrXInc= lumXInc>>1, chrDstW= dstW;

else					chrXInc= lumXInc,    chrDstW= dstW>>1;



if(dstbpp==12)	chrYInc= lumYInc,    chrDstH= dstH>>1;

else		chrYInc= lumYInc>>1, chrDstH= dstH;



  // force calculation of the horizontal interpolation of the first line



  if(srcSliceY ==0){

//	printf("dstW %d, srcw %d, mmx2 %d\n", dstW, srcW, canMMX2BeUsed);

	lumBufIndex=0;

	chrBufIndex=0;

	dstY=0;



	//precalculate horizontal scaler filter coefficients

	if(oldDstW!=dstW || oldSrcW!=srcW || oldFlags!=sws_flags)

	{

#ifdef HAVE_MMX

		const int filterAlign=4;

#else

		const int filterAlign=1;

#endif

		oldDstW= dstW; oldSrcW= srcW; oldFlags= sws_flags;



		if(sws_flags != SWS_FAST_BILINEAR)

		{

			RENAME(initFilter)(hLumFilter, hLumFilterPos, &hLumFilterSize, lumXInc,

					   srcW   , dstW   , filterAlign, 1<<14);

			RENAME(initFilter)(hChrFilter, hChrFilterPos, &hChrFilterSize, chrXInc,

					   srcW>>1, chrDstW, filterAlign, 1<<14);

		}



#ifdef HAVE_MMX2

// cant downscale !!!

		if(canMMX2BeUsed && sws_flags == SWS_FAST_BILINEAR)

		{

			initMMX2HScaler(dstW   , lumXInc, funnyYCode);

			initMMX2HScaler(chrDstW, chrXInc, funnyUVCode);

		}

#endif

	} // Init Horizontal stuff



	if(oldDstH!=dstH || oldSrcH!=srcH || oldFlags!=sws_flags)

	{

		int i;

		oldDstH= dstH; oldSrcH= srcH; oldFlags= sws_flags; //FIXME swsflags conflict with x check



		// deallocate pixbufs

		for(i=0; i<vLumBufSize; i++) free(lumPixBuf[i]);

		for(i=0; i<vChrBufSize; i++) free(chrPixBuf[i]);



		RENAME(initFilter)(vLumFilter, vLumFilterPos, &vLumFilterSize, lumYInc,

				srcH   , dstH,    1, (1<<12)-4);

		RENAME(initFilter)(vChrFilter, vChrFilterPos, &vChrFilterSize, chrYInc,

				srcH>>1, chrDstH, 1, (1<<12)-4);



		// Calculate Buffer Sizes so that they wont run out while handling these damn slices

		vLumBufSize= vLumFilterSize; vChrBufSize= vChrFilterSize;

		for(i=0; i<dstH; i++)

		{

			int chrI= i*chrDstH / dstH;

			int nextSlice= MAX(vLumFilterPos[i   ] + vLumFilterSize - 1,

					 ((vChrFilterPos[chrI] + vChrFilterSize - 1)<<1));

			nextSlice&= ~1; // Slices start at even boundaries

			if(vLumFilterPos[i   ] + vLumBufSize < nextSlice)

				vLumBufSize= nextSlice - vLumFilterPos[i   ];

			if(vChrFilterPos[chrI] + vChrBufSize < (nextSlice>>1))

				vChrBufSize= (nextSlice>>1) - vChrFilterPos[chrI];

		}



		// allocate pixbufs (we use dynamic allocation because otherwise we would need to

		// allocate several megabytes to handle all possible cases)

		for(i=0; i<vLumBufSize; i++)

			lumPixBuf[i]= lumPixBuf[i+vLumBufSize]= (uint16_t*)memalign(8, 4000);

		for(i=0; i<vChrBufSize; i++)

			chrPixBuf[i]= chrPixBuf[i+vChrBufSize]= (uint16_t*)memalign(8, 8000);



		//try to avoid drawing green stuff between the right end and the stride end

		for(i=0; i<vLumBufSize; i++) memset(lumPixBuf[i], 0, 4000);

		for(i=0; i<vChrBufSize; i++) memset(chrPixBuf[i], 64, 8000);



#ifdef HAVE_MMX

		// pack filter data for mmx code

		for(i=0; i<vLumFilterSize*dstH; i++)

			lumMmxFilter[4*i]=lumMmxFilter[4*i+1]=lumMmxFilter[4*i+2]=lumMmxFilter[4*i+3]=

				vLumFilter[i];



		for(i=0; i<vChrFilterSize*chrDstH; i++)

			chrMmxFilter[4*i]=chrMmxFilter[4*i+1]=chrMmxFilter[4*i+2]=chrMmxFilter[4*i+3]=

				vChrFilter[i];

#endif

	}



	lastInLumBuf= -1;

	lastInChrBuf= -1;

  } // if(firstLine)



	for(;dstY < dstH; dstY++){

		unsigned char *dest =dstptr[0]+dststride*dstY;

		unsigned char *uDest=dstptr[1]+(dststride>>1)*(dstY>>1);

		unsigned char *vDest=dstptr[2]+(dststride>>1)*(dstY>>1);

		const int chrDstY= dstbpp==12 ? (dstY>>1) : dstY;



		const int firstLumSrcY= vLumFilterPos[dstY]; //First line needed as input

		const int firstChrSrcY= vChrFilterPos[chrDstY]; //First line needed as input

		const int lastLumSrcY= firstLumSrcY + vLumFilterSize -1; // Last line needed as input

		const int lastChrSrcY= firstChrSrcY + vChrFilterSize -1; // Last line needed as input



		if(sws_flags == SWS_FAST_BILINEAR)

		{

			//handle holes

			if(firstLumSrcY > lastInLumBuf) lastInLumBuf= firstLumSrcY-1;

			if(firstChrSrcY > lastInChrBuf) lastInChrBuf= firstChrSrcY-1;

		}



		ASSERT(firstLumSrcY >= lastInLumBuf - vLumBufSize + 1)

		ASSERT(firstChrSrcY >= lastInChrBuf - vChrBufSize + 1)



		// Do we have enough lines in this slice to output the dstY line

		if(lastLumSrcY < srcSliceY + srcSliceH && lastChrSrcY < ((srcSliceY + srcSliceH)>>1))

		{

			//Do horizontal scaling

			while(lastInLumBuf < lastLumSrcY)

			{

				uint8_t *src= srcptr[0]+(lastInLumBuf + 1 - srcSliceY)*stride[0];

				lumBufIndex++;

				ASSERT(lumBufIndex < 2*vLumBufSize)

				ASSERT(lastInLumBuf + 1 - srcSliceY < srcSliceH)

				ASSERT(lastInLumBuf + 1 - srcSliceY >= 0)

//				printf("%d %d\n", lumBufIndex, vLumBufSize);

				RENAME(hyscale)(lumPixBuf[ lumBufIndex ], dstW, src, srcW, lumXInc);

				lastInLumBuf++;

			}

			while(lastInChrBuf < lastChrSrcY)

			{

				uint8_t *src1= srcptr[1]+(lastInChrBuf + 1 - (srcSliceY>>1))*stride[1];

				uint8_t *src2= srcptr[2]+(lastInChrBuf + 1 - (srcSliceY>>1))*stride[2];

				chrBufIndex++;

				ASSERT(chrBufIndex < 2*vChrBufSize)

				ASSERT(lastInChrBuf + 1 - (srcSliceY>>1) < (srcSliceH>>1))

				ASSERT(lastInChrBuf + 1 - (srcSliceY>>1) >= 0)

				RENAME(hcscale)(chrPixBuf[ chrBufIndex ], chrDstW, src1, src2, srcW>>1, chrXInc);

				lastInChrBuf++;

			}

			//wrap buf index around to stay inside the ring buffer

			if(lumBufIndex >= vLumBufSize ) lumBufIndex-= vLumBufSize;

			if(chrBufIndex >= vChrBufSize ) chrBufIndex-= vChrBufSize;

		}

		else // not enough lines left in this slice -> load the rest in the buffer

		{

/*		printf("%d %d Last:%d %d LastInBuf:%d %d Index:%d %d Y:%d FSize: %d %d BSize: %d %d\n",

			firstChrSrcY,firstLumSrcY,lastChrSrcY,lastLumSrcY,

			lastInChrBuf,lastInLumBuf,chrBufIndex,lumBufIndex,dstY,vChrFilterSize,vLumFilterSize,

			vChrBufSize, vLumBufSize);

*/

			//Do horizontal scaling

			while(lastInLumBuf+1 < srcSliceY + srcSliceH)

			{

				uint8_t *src= srcptr[0]+(lastInLumBuf + 1 - srcSliceY)*stride[0];

				lumBufIndex++;

				ASSERT(lumBufIndex < 2*vLumBufSize)

				ASSERT(lastInLumBuf + 1 - srcSliceY < srcSliceH)

				ASSERT(lastInLumBuf + 1 - srcSliceY >= 0)

				RENAME(hyscale)(lumPixBuf[ lumBufIndex ], dstW, src, srcW, lumXInc);

				lastInLumBuf++;

			}

			while(lastInChrBuf+1 < ((srcSliceY + srcSliceH)>>1))

			{

				uint8_t *src1= srcptr[1]+(lastInChrBuf + 1 - (srcSliceY>>1))*stride[1];

				uint8_t *src2= srcptr[2]+(lastInChrBuf + 1 - (srcSliceY>>1))*stride[2];

				chrBufIndex++;

				ASSERT(chrBufIndex < 2*vChrBufSize)

				ASSERT(lastInChrBuf + 1 - (srcSliceY>>1) < (srcSliceH>>1))

				ASSERT(lastInChrBuf + 1 - (srcSliceY>>1) >= 0)

				RENAME(hcscale)(chrPixBuf[ chrBufIndex ], chrDstW, src1, src2, srcW>>1, chrXInc);

				lastInChrBuf++;

			}

			//wrap buf index around to stay inside the ring buffer

			if(lumBufIndex >= vLumBufSize ) lumBufIndex-= vLumBufSize;

			if(chrBufIndex >= vChrBufSize ) chrBufIndex-= vChrBufSize;

			break; //we cant output a dstY line so lets try with the next slice

		}



#ifdef HAVE_MMX

		b5Dither= dither8[dstY&1];

		g6Dither= dither4[dstY&1];

		g5Dither= dither8[dstY&1];

		r5Dither= dither8[(dstY+1)&1];

#endif



		if(dstbpp==12) //YV12

		{

			if(dstY&1) uDest=vDest= NULL; //FIXME split functions in lumi / chromi

			if(vLumFilterSize == 1 && vChrFilterSize == 1) // Unscaled YV12

			{

				int16_t *lumBuf = lumPixBuf[0];

				int16_t *chrBuf= chrPixBuf[0];

				RENAME(yuv2yuv1)(lumBuf, chrBuf, dest, uDest, vDest, dstW);

			}

			else //General YV12

			{

				int16_t **lumSrcPtr= lumPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize;

				int16_t **chrSrcPtr= chrPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;

				RENAME(yuv2yuvX)(

					vLumFilter+dstY*vLumFilterSize     , lumSrcPtr, vLumFilterSize,

					vChrFilter+(dstY>>1)*vChrFilterSize, chrSrcPtr, vChrFilterSize,

					dest, uDest, vDest, dstW,

					lumMmxFilter+dstY*vLumFilterSize*4, chrMmxFilter+(dstY>>1)*vChrFilterSize*4);

			}

		}

		else

		{

			int16_t **lumSrcPtr= lumPixBuf + lumBufIndex + firstLumSrcY - lastInLumBuf + vLumBufSize;

			int16_t **chrSrcPtr= chrPixBuf + chrBufIndex + firstChrSrcY - lastInChrBuf + vChrBufSize;



			ASSERT(lumSrcPtr + vLumFilterSize - 1 < lumPixBuf + vLumBufSize*2);

			ASSERT(chrSrcPtr + vChrFilterSize - 1 < chrPixBuf + vChrBufSize*2);

			if(vLumFilterSize == 1 && vChrFilterSize == 2) //Unscaled RGB

			{

				int chrAlpha= vChrFilter[2*dstY+1];



				RENAME(yuv2rgb1)(*lumSrcPtr, *chrSrcPtr, *(chrSrcPtr+1),

						 dest, dstW, chrAlpha, dstbpp);

			}

			else if(vLumFilterSize == 2 && vChrFilterSize == 2) //BiLinear Upscale RGB

			{

				int lumAlpha= vLumFilter[2*dstY+1];

				int chrAlpha= vChrFilter[2*dstY+1];



				RENAME(yuv2rgb2)(*lumSrcPtr, *(lumSrcPtr+1), *chrSrcPtr, *(chrSrcPtr+1),

						 dest, dstW, lumAlpha, chrAlpha, dstbpp);

			}

			else //General RGB

			{

				RENAME(yuv2rgbX)(

					vLumFilter+dstY*vLumFilterSize, lumSrcPtr, vLumFilterSize,

					vChrFilter+dstY*vChrFilterSize, chrSrcPtr, vChrFilterSize,

					dest, dstW, dstbpp,

					lumMmxFilter+dstY*vLumFilterSize*4, chrMmxFilter+dstY*vChrFilterSize*4);

			}

		}

	}



#ifdef HAVE_MMX

	__asm __volatile(SFENCE:::"memory");

	__asm __volatile(EMMS:::"memory");

#endif

	firstTime=0;

}