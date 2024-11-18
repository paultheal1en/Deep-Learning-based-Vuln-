static void hybrid_synthesis(float out[2][38][64], float in[91][32][2], int is34, int len)

{

    int i, n;

    if (is34) {

        for (n = 0; n < len; n++) {

            memset(out[0][n], 0, 5*sizeof(out[0][n][0]));

            memset(out[1][n], 0, 5*sizeof(out[1][n][0]));

            for (i = 0; i < 12; i++) {

                out[0][n][0] += in[   i][n][0];

                out[1][n][0] += in[   i][n][1];

            }

            for (i = 0; i < 8; i++) {

                out[0][n][1] += in[12+i][n][0];

                out[1][n][1] += in[12+i][n][1];

            }

            for (i = 0; i < 4; i++) {

                out[0][n][2] += in[20+i][n][0];

                out[1][n][2] += in[20+i][n][1];

                out[0][n][3] += in[24+i][n][0];

                out[1][n][3] += in[24+i][n][1];

                out[0][n][4] += in[28+i][n][0];

                out[1][n][4] += in[28+i][n][1];

            }

        }

        for (i = 0; i < 59; i++) {

            for (n = 0; n < len; n++) {

                out[0][n][i+5] = in[i+32][n][0];

                out[1][n][i+5] = in[i+32][n][1];

            }

        }

    } else {

        for (n = 0; n < len; n++) {

            out[0][n][0] = in[0][n][0] + in[1][n][0] + in[2][n][0] +

                           in[3][n][0] + in[4][n][0] + in[5][n][0];

            out[1][n][0] = in[0][n][1] + in[1][n][1] + in[2][n][1] +

                           in[3][n][1] + in[4][n][1] + in[5][n][1];

            out[0][n][1] = in[6][n][0] + in[7][n][0];

            out[1][n][1] = in[6][n][1] + in[7][n][1];

            out[0][n][2] = in[8][n][0] + in[9][n][0];

            out[1][n][2] = in[8][n][1] + in[9][n][1];

        }

        for (i = 0; i < 61; i++) {

            for (n = 0; n < len; n++) {

                out[0][n][i+3] = in[i+10][n][0];

                out[1][n][i+3] = in[i+10][n][1];

            }

        }

    }

}
