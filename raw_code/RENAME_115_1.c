static inline void RENAME(yuv2packedX)(SwsContext *c, int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				    int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

			    uint8_t *dest, long dstW, long dstY)

{

#ifdef HAVE_MMX

    long dummy=0;

    if(c->flags & SWS_ACCURATE_RND){

                switch(c->dstFormat){

                case PIX_FMT_RGB32:

                                YSCALEYUV2PACKEDX_ACCURATE

				YSCALEYUV2RGBX

				WRITEBGR32(%4, %5, %%REGa)



                                YSCALEYUV2PACKEDX_END

                        return;

                case PIX_FMT_BGR24:

                                YSCALEYUV2PACKEDX_ACCURATE

				YSCALEYUV2RGBX

				"lea (%%"REG_a", %%"REG_a", 2), %%"REG_c"\n\t" //FIXME optimize

				"add %4, %%"REG_c"			\n\t"

				WRITEBGR24(%%REGc, %5, %%REGa)





			:: "r" (&c->redDither),

			   "m" (dummy), "m" (dummy), "m" (dummy),

			   "r" (dest), "m" (dstW)

			: "%"REG_a, "%"REG_c, "%"REG_d, "%"REG_S

			);

                        return;

                case PIX_FMT_BGR555:

                                YSCALEYUV2PACKEDX_ACCURATE

				YSCALEYUV2RGBX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

				"paddusb "MANGLE(b5Dither)", %%mm2\n\t"

				"paddusb "MANGLE(g5Dither)", %%mm4\n\t"

				"paddusb "MANGLE(r5Dither)", %%mm5\n\t"

#endif



				WRITEBGR15(%4, %5, %%REGa)

                                YSCALEYUV2PACKEDX_END

                        return;

                case PIX_FMT_BGR565:

                                YSCALEYUV2PACKEDX_ACCURATE

				YSCALEYUV2RGBX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

				"paddusb "MANGLE(b5Dither)", %%mm2\n\t"

				"paddusb "MANGLE(g6Dither)", %%mm4\n\t"

				"paddusb "MANGLE(r5Dither)", %%mm5\n\t"

#endif



				WRITEBGR16(%4, %5, %%REGa)

                                YSCALEYUV2PACKEDX_END

                        return;

                case PIX_FMT_YUYV422:

				YSCALEYUV2PACKEDX_ACCURATE

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */



				"psraw $3, %%mm3		\n\t"

				"psraw $3, %%mm4		\n\t"

				"psraw $3, %%mm1		\n\t"

				"psraw $3, %%mm7		\n\t"

				WRITEYUY2(%4, %5, %%REGa)

                                YSCALEYUV2PACKEDX_END

                        return;

                }

    }else{

	switch(c->dstFormat)

	{

	case PIX_FMT_RGB32:

                                YSCALEYUV2PACKEDX

				YSCALEYUV2RGBX

				WRITEBGR32(%4, %5, %%REGa)

                                YSCALEYUV2PACKEDX_END

		return;

	case PIX_FMT_BGR24:

                                YSCALEYUV2PACKEDX

				YSCALEYUV2RGBX

				"lea (%%"REG_a", %%"REG_a", 2), %%"REG_c"\n\t" //FIXME optimize

				"add %4, %%"REG_c"			\n\t"

				WRITEBGR24(%%REGc, %5, %%REGa)



			:: "r" (&c->redDither),

			   "m" (dummy), "m" (dummy), "m" (dummy),

			   "r" (dest), "m" (dstW)

			: "%"REG_a, "%"REG_c, "%"REG_d, "%"REG_S

			);

		return;

	case PIX_FMT_BGR555:

                                YSCALEYUV2PACKEDX

				YSCALEYUV2RGBX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

				"paddusb "MANGLE(b5Dither)", %%mm2\n\t"

				"paddusb "MANGLE(g5Dither)", %%mm4\n\t"

				"paddusb "MANGLE(r5Dither)", %%mm5\n\t"

#endif



				WRITEBGR15(%4, %5, %%REGa)

                                YSCALEYUV2PACKEDX_END

		return;

	case PIX_FMT_BGR565:

                                YSCALEYUV2PACKEDX

				YSCALEYUV2RGBX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */

#ifdef DITHER1XBPP

				"paddusb "MANGLE(b5Dither)", %%mm2\n\t"

				"paddusb "MANGLE(g6Dither)", %%mm4\n\t"

				"paddusb "MANGLE(r5Dither)", %%mm5\n\t"

#endif



				WRITEBGR16(%4, %5, %%REGa)

                                YSCALEYUV2PACKEDX_END

		return;

	case PIX_FMT_YUYV422:

				YSCALEYUV2PACKEDX

		/* mm2=B, %%mm4=G, %%mm5=R, %%mm7=0 */



				"psraw $3, %%mm3		\n\t"

				"psraw $3, %%mm4		\n\t"

				"psraw $3, %%mm1		\n\t"

				"psraw $3, %%mm7		\n\t"

				WRITEYUY2(%4, %5, %%REGa)

                                YSCALEYUV2PACKEDX_END

		return;

        }

    }

#endif

#ifdef HAVE_ALTIVEC

		/* The following list of supported dstFormat values should

		   match what's found in the body of altivec_yuv2packedX() */

		if(c->dstFormat==PIX_FMT_ABGR  || c->dstFormat==PIX_FMT_BGRA  ||

		   c->dstFormat==PIX_FMT_BGR24 || c->dstFormat==PIX_FMT_RGB24 ||

		   c->dstFormat==PIX_FMT_RGBA  || c->dstFormat==PIX_FMT_ARGB)

			altivec_yuv2packedX (c, lumFilter, lumSrc, lumFilterSize,

				    chrFilter, chrSrc, chrFilterSize,

				    dest, dstW, dstY);

		else

#endif

			yuv2packedXinC(c, lumFilter, lumSrc, lumFilterSize,

				    chrFilter, chrSrc, chrFilterSize,

				    dest, dstW, dstY);

}
