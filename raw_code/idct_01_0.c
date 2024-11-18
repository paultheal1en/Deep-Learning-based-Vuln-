static void idct(uint8_t *dst, int dst_linesize, int src[64])

{

    int i, j, k;

    double tmp[64];



    for (i = 0; i < 8; i++) {

        for (j = 0; j < 8; j++) {

            double sum = 0.0;



            for (k = 0; k < 8; k++)

                sum += c[k*8+j] * src[8*i+k];



            tmp[8*i+j] = sum;

        }

    }



    for (j = 0; j < 8; j++) {

        for (i = 0; i < 8; i++) {

            double sum = 0.0;



            for (k = 0; k < 8; k++)

                sum += c[k*8+i]*tmp[8*k+j];



            dst[dst_linesize*i + j] = av_clip_uint8((int)floor(sum+0.5));

        }

    }

}
