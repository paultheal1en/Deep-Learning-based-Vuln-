altivec_yuv2packedX (SwsContext *c,

		       int16_t *lumFilter, int16_t **lumSrc, int lumFilterSize,

		       int16_t *chrFilter, int16_t **chrSrc, int chrFilterSize,

		       uint8_t *dest, int dstW, int dstY)

{

  int i,j;

  short tmp __attribute__((aligned (16)));

  int16_t *p;

  short *f;

  vector signed short X,X0,X1,Y0,U0,V0,Y1,U1,V1,U,V;

  vector signed short R0,G0,B0,R1,G1,B1;



  vector unsigned char R,G,B,pels[3];

  vector unsigned char *out,*nout;



  vector signed short   RND = vec_splat((vector signed short)AVV(1<<3),0);

  vector unsigned short SCL = vec_splat((vector unsigned short)AVV(4),0);

  unsigned long scratch[16] __attribute__ ((aligned (16)));



  vector signed short *vYCoeffsBank, *vCCoeffsBank;



  vector signed short *YCoeffs, *CCoeffs;



  vYCoeffsBank = malloc (sizeof (vector signed short)*lumFilterSize*dstW);

  vCCoeffsBank = malloc (sizeof (vector signed short)*chrFilterSize*dstW);



  for (i=0;i<lumFilterSize*dstW;i++) {

    tmp = c->vLumFilter[i];

    p = &vYCoeffsBank[i];

    for (j=0;j<8;j++)

      p[j] = tmp;

  }



  for (i=0;i<chrFilterSize*dstW;i++) {

    tmp = c->vChrFilter[i];

    p = &vCCoeffsBank[i];

    for (j=0;j<8;j++)

      p[j] = tmp;

  }



  YCoeffs = vYCoeffsBank+dstY*lumFilterSize;

  CCoeffs = vCCoeffsBank+dstY*chrFilterSize;



  out = (vector unsigned char *)dest;



  for(i=0; i<dstW; i+=16){

    Y0 = RND;

    Y1 = RND;

    /* extract 16 coeffs from lumSrc */

    for(j=0; j<lumFilterSize; j++) {

      X0 = vec_ld (0,  &lumSrc[j][i]);

      X1 = vec_ld (16, &lumSrc[j][i]);

      Y0 = vec_mradds (X0, YCoeffs[j], Y0);

      Y1 = vec_mradds (X1, YCoeffs[j], Y1);

    }



    U = RND;

    V = RND;

    /* extract 8 coeffs from U,V */

    for(j=0; j<chrFilterSize; j++) {

      X  = vec_ld (0, &chrSrc[j][i/2]);

      U  = vec_mradds (X, CCoeffs[j], U);

      X  = vec_ld (0, &chrSrc[j][i/2+2048]);

      V  = vec_mradds (X, CCoeffs[j], V);

    }



    /* scale and clip signals */

    Y0 = vec_sra (Y0, SCL);

    Y1 = vec_sra (Y1, SCL);

    U  = vec_sra (U,  SCL);

    V  = vec_sra (V,  SCL);



    Y0 = vec_clip (Y0);

    Y1 = vec_clip (Y1);

    U  = vec_clip (U);

    V  = vec_clip (V);



    /* now we have

      Y0= y0 y1 y2 y3 y4 y5 y6 y7     Y1= y8 y9 y10 y11 y12 y13 y14 y15

      U= u0 u1 u2 u3 u4 u5 u6 u7      V= v0 v1 v2 v3 v4 v5 v6 v7



      Y0= y0 y1 y2 y3 y4 y5 y6 y7    Y1= y8 y9 y10 y11 y12 y13 y14 y15

      U0= u0 u0 u1 u1 u2 u2 u3 u3    U1= u4 u4 u5 u5 u6 u6 u7 u7

      V0= v0 v0 v1 v1 v2 v2 v3 v3    V1= v4 v4 v5 v5 v6 v6 v7 v7

    */



    U0 = vec_mergeh (U,U);

    V0 = vec_mergeh (V,V);



    U1 = vec_mergel (U,U);

    V1 = vec_mergel (V,V);



    cvtyuvtoRGB (c, Y0,U0,V0,&R0,&G0,&B0);

    cvtyuvtoRGB (c, Y1,U1,V1,&R1,&G1,&B1);



    R  = vec_packclp (R0,R1);

    G  = vec_packclp (G0,G1);

    B  = vec_packclp (B0,B1);



    out_rgba (R,G,B,out);

  }



  if (i < dstW) {

    i -= 16;



    Y0 = RND;

    Y1 = RND;

    /* extract 16 coeffs from lumSrc */

    for(j=0; j<lumFilterSize; j++) {

      X0 = vec_ld (0,  &lumSrc[j][i]);

      X1 = vec_ld (16, &lumSrc[j][i]);

      Y0 = vec_mradds (X0, YCoeffs[j], Y0);

      Y1 = vec_mradds (X1, YCoeffs[j], Y1);

    }



    U = RND;

    V = RND;

    /* extract 8 coeffs from U,V */

    for(j=0; j<chrFilterSize; j++) {

      X  = vec_ld (0, &chrSrc[j][i/2]);

      U  = vec_mradds (X, CCoeffs[j], U);

      X  = vec_ld (0, &chrSrc[j][i/2+2048]);

      V  = vec_mradds (X, CCoeffs[j], V);

    }



    /* scale and clip signals */

    Y0 = vec_sra (Y0, SCL);

    Y1 = vec_sra (Y1, SCL);

    U  = vec_sra (U,  SCL);

    V  = vec_sra (V,  SCL);



    Y0 = vec_clip (Y0);

    Y1 = vec_clip (Y1);

    U  = vec_clip (U);

    V  = vec_clip (V);



    /* now we have

       Y0= y0 y1 y2 y3 y4 y5 y6 y7     Y1= y8 y9 y10 y11 y12 y13 y14 y15

       U= u0 u1 u2 u3 u4 u5 u6 u7      V= v0 v1 v2 v3 v4 v5 v6 v7



       Y0= y0 y1 y2 y3 y4 y5 y6 y7    Y1= y8 y9 y10 y11 y12 y13 y14 y15

       U0= u0 u0 u1 u1 u2 u2 u3 u3    U1= u4 u4 u5 u5 u6 u6 u7 u7

       V0= v0 v0 v1 v1 v2 v2 v3 v3    V1= v4 v4 v5 v5 v6 v6 v7 v7

    */



    U0 = vec_mergeh (U,U);

    V0 = vec_mergeh (V,V);



    U1 = vec_mergel (U,U);

    V1 = vec_mergel (V,V);



    cvtyuvtoRGB (c, Y0,U0,V0,&R0,&G0,&B0);

    cvtyuvtoRGB (c, Y1,U1,V1,&R1,&G1,&B1);



    R  = vec_packclp (R0,R1);

    G  = vec_packclp (G0,G1);

    B  = vec_packclp (B0,B1);



    nout = (vector unsigned char *)scratch;

    out_rgba (R,G,B,nout);



    memcpy (&((uint32_t*)dest)[i], scratch, (dstW-i)/4);

  }



  if (vYCoeffsBank) free (vYCoeffsBank);

  if (vCCoeffsBank) free (vCCoeffsBank);



}
