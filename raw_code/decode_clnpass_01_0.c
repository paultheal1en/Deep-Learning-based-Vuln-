static void decode_clnpass(Jpeg2000DecoderContext *s, Jpeg2000T1Context *t1,

                           int width, int height, int bpno, int bandno,

                           int seg_symbols)

{

    int mask = 3 << (bpno - 1), y0, x, y, runlen, dec;



    for (y0 = 0; y0 < height; y0 += 4)

        for (x = 0; x < width; x++) {

            if (y0 + 3 < height &&

                !((t1->flags[y0 + 1][x + 1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG)) ||

                  (t1->flags[y0 + 2][x + 1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG)) ||

                  (t1->flags[y0 + 3][x + 1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG)) ||

                  (t1->flags[y0 + 4][x + 1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG)))) {

                if (!ff_mqc_decode(&t1->mqc, t1->mqc.cx_states + MQC_CX_RL))

                    continue;

                runlen = ff_mqc_decode(&t1->mqc,

                                       t1->mqc.cx_states + MQC_CX_UNI);

                runlen = (runlen << 1) | ff_mqc_decode(&t1->mqc,

                                                       t1->mqc.cx_states +

                                                       MQC_CX_UNI);

                dec = 1;

            } else {

                runlen = 0;

                dec    = 0;

            }



            for (y = y0 + runlen; y < y0 + 4 && y < height; y++) {

                if (!dec) {

                    if (!(t1->flags[y + 1][x + 1] & (JPEG2000_T1_SIG | JPEG2000_T1_VIS)))

                        dec = ff_mqc_decode(&t1->mqc,

                                            t1->mqc.cx_states +

                                            ff_jpeg2000_getsigctxno(t1->flags[y + 1][x + 1],

                                                                   bandno));

                }

                if (dec) {

                    int xorbit;

                    int ctxno = ff_jpeg2000_getsgnctxno(t1->flags[y + 1][x + 1],

                                                        &xorbit);

                    t1->data[y][x] = (ff_mqc_decode(&t1->mqc,

                                                    t1->mqc.cx_states + ctxno) ^

                                      xorbit)

                                     ? -mask : mask;

                    ff_jpeg2000_set_significance(t1, x, y, t1->data[y][x] < 0);

                }

                dec = 0;

                t1->flags[y + 1][x + 1] &= ~JPEG2000_T1_VIS;

            }

        }

    if (seg_symbols) {

        int val;

        val = ff_mqc_decode(&t1->mqc, t1->mqc.cx_states + MQC_CX_UNI);

        val = (val << 1) + ff_mqc_decode(&t1->mqc, t1->mqc.cx_states + MQC_CX_UNI);

        val = (val << 1) + ff_mqc_decode(&t1->mqc, t1->mqc.cx_states + MQC_CX_UNI);

        val = (val << 1) + ff_mqc_decode(&t1->mqc, t1->mqc.cx_states + MQC_CX_UNI);

        if (val != 0xa)

            av_log(s->avctx, AV_LOG_ERROR,

                   "Segmentation symbol value incorrect\n");

    }

}
