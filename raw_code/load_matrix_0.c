static int load_matrix(MpegEncContext *s, uint16_t matrix0[64], uint16_t matrix1[64], int intra)

{

    int i;



    for (i = 0; i < 64; i++) {

        int j = s->dsp.idct_permutation[ff_zigzag_direct[i]];

        int v = get_bits(&s->gb, 8);

        if (v == 0) {

            av_log(s->avctx, AV_LOG_ERROR, "matrix damaged\n");

            return -1;

        }

        if (intra && i == 0 && v != 8) {

            av_log(s->avctx, AV_LOG_ERROR, "intra matrix specifies invalid DC quantizer %d, ignoring\n", v);

            v = 8; // needed by pink.mpg / issue1046

        }

        matrix0[j] = v;

        if (matrix1)

            matrix1[j] = v;

    }

    return 0;

}
