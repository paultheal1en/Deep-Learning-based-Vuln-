static void hybrid6_cx(float (*in)[2], float (*out)[32][2], const float (*filter)[7][2], int len)

{

    int i, j, ssb;

    int N = 8;

    float temp[8][2];



    for (i = 0; i < len; i++, in++) {

        for (ssb = 0; ssb < N; ssb++) {

            float sum_re = filter[ssb][6][0] * in[6][0], sum_im = filter[ssb][6][0] * in[6][1];

            for (j = 0; j < 6; j++) {

                float in0_re = in[j][0];

                float in0_im = in[j][1];

                float in1_re = in[12-j][0];

                float in1_im = in[12-j][1];

                sum_re += filter[ssb][j][0] * (in0_re + in1_re) - filter[ssb][j][1] * (in0_im - in1_im);

                sum_im += filter[ssb][j][0] * (in0_im + in1_im) + filter[ssb][j][1] * (in0_re - in1_re);

            }

            temp[ssb][0] = sum_re;

            temp[ssb][1] = sum_im;

        }

        out[0][i][0] = temp[6][0];

        out[0][i][1] = temp[6][1];

        out[1][i][0] = temp[7][0];

        out[1][i][1] = temp[7][1];

        out[2][i][0] = temp[0][0];

        out[2][i][1] = temp[0][1];

        out[3][i][0] = temp[1][0];

        out[3][i][1] = temp[1][1];

        out[4][i][0] = temp[2][0] + temp[5][0];

        out[4][i][1] = temp[2][1] + temp[5][1];

        out[5][i][0] = temp[3][0] + temp[4][0];

        out[5][i][1] = temp[3][1] + temp[4][1];

    }

}
