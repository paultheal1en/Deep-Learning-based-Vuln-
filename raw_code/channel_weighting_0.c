static void channel_weighting(float *su1, float *su2, int *p3)

{

    int band, nsample;

    /* w[x][y] y=0 is left y=1 is right */

    float w[2][2];



    if (p3[1] != 7 || p3[3] != 7) {

        get_channel_weights(p3[1], p3[0], w[0]);

        get_channel_weights(p3[3], p3[2], w[1]);



        for (band = 1; band < 4; band++) {

            for (nsample = 0; nsample < 8; nsample++) {

                su1[band * 256 + nsample] *= INTERPOLATE(w[0][0], w[0][1], nsample);

                su2[band * 256 + nsample] *= INTERPOLATE(w[1][0], w[1][1], nsample);

            }

            for(; nsample < 256; nsample++) {

                su1[band * 256  + nsample] *= w[1][0];

                su2[band * 256  + nsample] *= w[1][1];

            }

        }

    }

}
