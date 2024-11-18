static void hybrid_analysis(float out[91][32][2], float in[5][44][2], float L[2][38][64], int is34, int len)

{

    int i, j;

    for (i = 0; i < 5; i++) {

        for (j = 0; j < 38; j++) {

            in[i][j+6][0] = L[0][j][i];

            in[i][j+6][1] = L[1][j][i];

        }

    }

    if (is34) {

        hybrid4_8_12_cx(in[0], out,    f34_0_12, 12, len);

        hybrid4_8_12_cx(in[1], out+12, f34_1_8,   8, len);

        hybrid4_8_12_cx(in[2], out+20, f34_2_4,   4, len);

        hybrid4_8_12_cx(in[3], out+24, f34_2_4,   4, len);

        hybrid4_8_12_cx(in[4], out+28, f34_2_4,   4, len);

        for (i = 0; i < 59; i++) {

            for (j = 0; j < len; j++) {

                out[i+32][j][0] = L[0][j][i+5];

                out[i+32][j][1] = L[1][j][i+5];

            }

        }

    } else {

        hybrid6_cx(in[0], out, f20_0_8, len);

        hybrid2_re(in[1], out+6, g1_Q2, len, 1);

        hybrid2_re(in[2], out+8, g1_Q2, len, 0);

        for (i = 0; i < 61; i++) {

            for (j = 0; j < len; j++) {

                out[i+10][j][0] = L[0][j][i+3];

                out[i+10][j][1] = L[1][j][i+3];

            }

        }

    }

    //update in_buf

    for (i = 0; i < 5; i++) {

        memcpy(in[i], in[i]+32, 6 * sizeof(in[i][0]));

    }

}
