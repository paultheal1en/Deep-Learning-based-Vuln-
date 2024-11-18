static void encode_clnpass(Jpeg2000T1Context *t1, int width, int height, int bandno, int *nmsedec, int bpno)

{

    int y0, x, y, mask = 1 << (bpno + NMSEDEC_FRACBITS);

    for (y0 = 0; y0 < height; y0 += 4)

        for (x = 0; x < width; x++){

            if (y0 + 3 < height && !(

            (t1->flags[y0+1][x+1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG)) ||

            (t1->flags[y0+2][x+1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG)) ||

            (t1->flags[y0+3][x+1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG)) ||

            (t1->flags[y0+4][x+1] & (JPEG2000_T1_SIG_NB | JPEG2000_T1_VIS | JPEG2000_T1_SIG))))

            {

                // aggregation mode

                int rlen;

                for (rlen = 0; rlen < 4; rlen++)

                    if (t1->data[y0+rlen][x] & mask)

                        break;

                ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + MQC_CX_RL, rlen != 4);

                if (rlen == 4)

                    continue;

                ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + MQC_CX_UNI, rlen >> 1);

                ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + MQC_CX_UNI, rlen & 1);

                for (y = y0 + rlen; y < y0 + 4; y++){

                    if (!(t1->flags[y+1][x+1] & (JPEG2000_T1_SIG | JPEG2000_T1_VIS))){

                        int ctxno = ff_jpeg2000_getsigctxno(t1->flags[y+1][x+1], bandno);

                        if (y > y0 + rlen)

                            ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + ctxno, t1->data[y][x] & mask ? 1:0);

                        if (t1->data[y][x] & mask){ // newly significant

                            int xorbit;

                            int ctxno = ff_jpeg2000_getsgnctxno(t1->flags[y+1][x+1], &xorbit);

                            *nmsedec += getnmsedec_sig(t1->data[y][x], bpno + NMSEDEC_FRACBITS);

                            ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + ctxno, (t1->flags[y+1][x+1] >> 15) ^ xorbit);

                            ff_jpeg2000_set_significance(t1, x, y, t1->flags[y+1][x+1] >> 15);

                        }

                    }

                    t1->flags[y+1][x+1] &= ~JPEG2000_T1_VIS;

                }

            } else{

                for (y = y0; y < y0 + 4 && y < height; y++){

                    if (!(t1->flags[y+1][x+1] & (JPEG2000_T1_SIG | JPEG2000_T1_VIS))){

                        int ctxno = ff_jpeg2000_getsigctxno(t1->flags[y+1][x+1], bandno);

                        ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + ctxno, t1->data[y][x] & mask ? 1:0);

                        if (t1->data[y][x] & mask){ // newly significant

                            int xorbit;

                            int ctxno = ff_jpeg2000_getsgnctxno(t1->flags[y+1][x+1], &xorbit);

                            *nmsedec += getnmsedec_sig(t1->data[y][x], bpno + NMSEDEC_FRACBITS);

                            ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + ctxno, (t1->flags[y+1][x+1] >> 15) ^ xorbit);

                            ff_jpeg2000_set_significance(t1, x, y, t1->flags[y+1][x+1] >> 15);

                        }

                    }

                    t1->flags[y+1][x+1] &= ~JPEG2000_T1_VIS;

                }

            }

        }

}
