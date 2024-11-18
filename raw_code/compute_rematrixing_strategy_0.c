static void compute_rematrixing_strategy(AC3EncodeContext *s)

{

    int nb_coefs;

    int blk, bnd, i;

    AC3Block *block, *av_uninit(block0);



    if (s->channel_mode != AC3_CHMODE_STEREO)

        return;



    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {

        block = &s->blocks[blk];

        block->new_rematrixing_strategy = !blk;



        if (!s->rematrixing_enabled) {

            block0 = block;

            continue;

        }



        block->num_rematrixing_bands = 4;

        if (block->cpl_in_use) {

            block->num_rematrixing_bands -= (s->start_freq[CPL_CH] <= 61);

            block->num_rematrixing_bands -= (s->start_freq[CPL_CH] == 37);

            if (blk && block->num_rematrixing_bands != block0->num_rematrixing_bands)

                block->new_rematrixing_strategy = 1;

        }

        nb_coefs = FFMIN(block->end_freq[1], block->end_freq[2]);



        for (bnd = 0; bnd < block->num_rematrixing_bands; bnd++) {

            /* calculate calculate sum of squared coeffs for one band in one block */

            int start = ff_ac3_rematrix_band_tab[bnd];

            int end   = FFMIN(nb_coefs, ff_ac3_rematrix_band_tab[bnd+1]);

            CoefSumType sum[4] = {0,};

            for (i = start; i < end; i++) {

                CoefType lt = block->mdct_coef[1][i];

                CoefType rt = block->mdct_coef[2][i];

                CoefType md = lt + rt;

                CoefType sd = lt - rt;

                MAC_COEF(sum[0], lt, lt);

                MAC_COEF(sum[1], rt, rt);

                MAC_COEF(sum[2], md, md);

                MAC_COEF(sum[3], sd, sd);

            }



            /* compare sums to determine if rematrixing will be used for this band */

            if (FFMIN(sum[2], sum[3]) < FFMIN(sum[0], sum[1]))

                block->rematrixing_flags[bnd] = 1;

            else

                block->rematrixing_flags[bnd] = 0;



            /* determine if new rematrixing flags will be sent */

            if (blk &&

                block->rematrixing_flags[bnd] != block0->rematrixing_flags[bnd]) {

                block->new_rematrixing_strategy = 1;

            }

        }

        block0 = block;

    }

}
