static void extract_exponents(AC3EncodeContext *s)

{

    int blk, ch, i;



    for (ch = 0; ch < s->channels; ch++) {

        for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {

            AC3Block *block = &s->blocks[blk];

            for (i = 0; i < AC3_MAX_COEFS; i++) {

                int e;

                int v = abs(SCALE_COEF(block->mdct_coef[ch][i]));

                if (v == 0)

                    e = 24;

                else {

                    e = 23 - av_log2(v) + block->exp_shift[ch];

                    if (e >= 24) {

                        e = 24;

                        block->mdct_coef[ch][i] = 0;

                    }

                }

                block->exp[ch][i] = e;

            }

        }

    }

}
