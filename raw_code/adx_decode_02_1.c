static void adx_decode(short *out,const unsigned char *in,PREV *prev)

{

	int scale = ((in[0]<<8)|(in[1]));

	int i;

	int s0,s1,s2,d;



//	printf("%x ",scale);



	in+=2;

	s1 = prev->s1;

	s2 = prev->s2;

	for(i=0;i<16;i++) {

		d = in[i];

		// d>>=4; if (d&8) d-=16;

		d = ((signed char)d >> 4);

		s0 = (BASEVOL*d*scale + SCALE1*s1 - SCALE2*s2)>>14;

		CLIP(s0);

		*out++=s0;

		s2 = s1;

		s1 = s0;



		d = in[i];

		//d&=15; if (d&8) d-=16;

		d = ((signed char)(d<<4) >> 4);

		s0 = (BASEVOL*d*scale + SCALE1*s1 - SCALE2*s2)>>14;

		CLIP(s0);

		*out++=s0;

		s2 = s1;

		s1 = s0;

	}

	prev->s1 = s1;

	prev->s2 = s2;



}
