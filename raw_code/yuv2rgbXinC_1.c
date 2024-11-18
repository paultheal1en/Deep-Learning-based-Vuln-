static inline void yuv2rgbXinC(int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

				    int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

				    uint8_t *dest, int dstW, int dstbpp)

{

	if(dstbpp==32)

	{

		int i;

		for(i=0; i<(dstW>>1); i++){

			int j;

			int Y1=0;

			int Y2=0;

			int U=0;

			int V=0;

			int Cb, Cr, Cg;

			for(j=0; j<lumFilterSize; j++)

			{

				Y1 += lumSrc[j][2*i] * lumFilter[j];

				Y2 += lumSrc[j][2*i+1] * lumFilter[j];

			}

			for(j=0; j<chrFilterSize; j++)

			{

				U += chrSrc[j][i] * chrFilter[j];

				V += chrSrc[j][i+2048] * chrFilter[j];

			}

			Y1= clip_yuvtab_2568[ (Y1>>19) + 256 ];

			Y2= clip_yuvtab_2568[ (Y2>>19) + 256 ];

			U >>= 19;

			V >>= 19;



			Cb= clip_yuvtab_40cf[U+ 256];

			Cg= clip_yuvtab_1a1e[V+ 256] + yuvtab_0c92[U+ 256];

			Cr= clip_yuvtab_3343[V+ 256];



			dest[8*i+0]=clip_table[((Y1 + Cb) >>13)];

			dest[8*i+1]=clip_table[((Y1 + Cg) >>13)];

			dest[8*i+2]=clip_table[((Y1 + Cr) >>13)];



			dest[8*i+4]=clip_table[((Y2 + Cb) >>13)];

			dest[8*i+5]=clip_table[((Y2 + Cg) >>13)];

			dest[8*i+6]=clip_table[((Y2 + Cr) >>13)];

		}

	}

	else if(dstbpp==24)

	{

		int i;

		for(i=0; i<(dstW>>1); i++){

			int j;

			int Y1=0;

			int Y2=0;

			int U=0;

			int V=0;

			int Cb, Cr, Cg;

			for(j=0; j<lumFilterSize; j++)

			{

				Y1 += lumSrc[j][2*i] * lumFilter[j];

				Y2 += lumSrc[j][2*i+1] * lumFilter[j];

			}

			for(j=0; j<chrFilterSize; j++)

			{

				U += chrSrc[j][i] * chrFilter[j];

				V += chrSrc[j][i+2048] * chrFilter[j];

			}

			Y1= clip_yuvtab_2568[ (Y1>>19) + 256 ];

			Y2= clip_yuvtab_2568[ (Y2>>19) + 256 ];

			U >>= 19;

			V >>= 19;



			Cb= clip_yuvtab_40cf[U+ 256];

			Cg= clip_yuvtab_1a1e[V+ 256] + yuvtab_0c92[U+ 256];

			Cr= clip_yuvtab_3343[V+ 256];



			dest[0]=clip_table[((Y1 + Cb) >>13)];

			dest[1]=clip_table[((Y1 + Cg) >>13)];

			dest[2]=clip_table[((Y1 + Cr) >>13)];



			dest[3]=clip_table[((Y2 + Cb) >>13)];

			dest[4]=clip_table[((Y2 + Cg) >>13)];

			dest[5]=clip_table[((Y2 + Cr) >>13)];

			dest+=6;

		}

	}

	else if(dstbpp==16)

	{

		int i;

		for(i=0; i<(dstW>>1); i++){

			int j;

			int Y1=0;

			int Y2=0;

			int U=0;

			int V=0;

			int Cb, Cr, Cg;

			for(j=0; j<lumFilterSize; j++)

			{

				Y1 += lumSrc[j][2*i] * lumFilter[j];

				Y2 += lumSrc[j][2*i+1] * lumFilter[j];

			}

			for(j=0; j<chrFilterSize; j++)

			{

				U += chrSrc[j][i] * chrFilter[j];

				V += chrSrc[j][i+2048] * chrFilter[j];

			}

			Y1= clip_yuvtab_2568[ (Y1>>19) + 256 ];

			Y2= clip_yuvtab_2568[ (Y2>>19) + 256 ];

			U >>= 19;

			V >>= 19;



			Cb= clip_yuvtab_40cf[U+ 256];

			Cg= clip_yuvtab_1a1e[V+ 256] + yuvtab_0c92[U+ 256];

			Cr= clip_yuvtab_3343[V+ 256];



			((uint16_t*)dest)[2*i] =

				clip_table16b[(Y1 + Cb) >>13] |

				clip_table16g[(Y1 + Cg) >>13] |

				clip_table16r[(Y1 + Cr) >>13];



			((uint16_t*)dest)[2*i+1] =

				clip_table16b[(Y2 + Cb) >>13] |

				clip_table16g[(Y2 + Cg) >>13] |

				clip_table16r[(Y2 + Cr) >>13];

		}

	}

	else if(dstbpp==15)

	{

		int i;

		for(i=0; i<(dstW>>1); i++){

			int j;

			int Y1=0;

			int Y2=0;

			int U=0;

			int V=0;

			int Cb, Cr, Cg;

			for(j=0; j<lumFilterSize; j++)

			{

				Y1 += lumSrc[j][2*i] * lumFilter[j];

				Y2 += lumSrc[j][2*i+1] * lumFilter[j];

			}

			for(j=0; j<chrFilterSize; j++)

			{

				U += chrSrc[j][i] * chrFilter[j];

				V += chrSrc[j][i+2048] * chrFilter[j];

			}

			Y1= clip_yuvtab_2568[ (Y1>>19) + 256 ];

			Y2= clip_yuvtab_2568[ (Y2>>19) + 256 ];

			U >>= 19;

			V >>= 19;



			Cb= clip_yuvtab_40cf[U+ 256];

			Cg= clip_yuvtab_1a1e[V+ 256] + yuvtab_0c92[U+ 256];

			Cr= clip_yuvtab_3343[V+ 256];



			((uint16_t*)dest)[2*i] =

				clip_table15b[(Y1 + Cb) >>13] |

				clip_table15g[(Y1 + Cg) >>13] |

				clip_table15r[(Y1 + Cr) >>13];



			((uint16_t*)dest)[2*i+1] =

				clip_table15b[(Y2 + Cb) >>13] |

				clip_table15g[(Y2 + Cg) >>13] |

				clip_table15r[(Y2 + Cr) >>13];

		}

	}

}
