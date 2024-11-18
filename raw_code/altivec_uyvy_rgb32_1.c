static int altivec_uyvy_rgb32 (SwsContext *c,

			       unsigned char **in, int *instrides,

			       int srcSliceY,	int srcSliceH,

			       unsigned char **oplanes, int *outstrides)

{

  int w = c->srcW;

  int h = srcSliceH;

  int i,j;

  vector unsigned char uyvy;

  vector signed   short Y,U,V;

  vector signed   short R0,G0,B0,R1,G1,B1;

  vector unsigned char  R,G,B;

  vector unsigned char *out;

  ubyte *img;



  img = in[0];

  out = (vector unsigned char *)(oplanes[0]+srcSliceY*outstrides[0]);



  for (i=0;i<h;i++) {

    for (j=0;j<w/16;j++) {

      uyvy = vec_ld (0, img);

      U = (vector signed short)

	vec_perm (uyvy, (vector unsigned char)AVV(0), demux_u);



      V = (vector signed short)

	vec_perm (uyvy, (vector unsigned char)AVV(0), demux_v);



      Y = (vector signed short)

	vec_perm (uyvy, (vector unsigned char)AVV(0), demux_y);



      cvtyuvtoRGB (c, Y,U,V,&R0,&G0,&B0);



      uyvy = vec_ld (16, img);

      U = (vector signed short)

	vec_perm (uyvy, (vector unsigned char)AVV(0), demux_u);



      V = (vector signed short)

	vec_perm (uyvy, (vector unsigned char)AVV(0), demux_v);



      Y = (vector signed short)

	vec_perm (uyvy, (vector unsigned char)AVV(0), demux_y);



      cvtyuvtoRGB (c, Y,U,V,&R1,&G1,&B1);



      R  = vec_packclp (R0,R1);

      G  = vec_packclp (G0,G1);

      B  = vec_packclp (B0,B1);



      //      vec_mstbgr24 (R,G,B, out);

      out_rgba (R,G,B,out);



      img += 32;

    }

  }

  return srcSliceH;

}
