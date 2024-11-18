static int decode_block_coeffs(VP56RangeCoder *c, DCTELEM block[16],

                               uint8_t probs[8][3][NUM_DCT_TOKENS-1],

                               int i, int zero_nhood, int16_t qmul[2])

{

    uint8_t *token_prob;

    int nonzero = 0;

    int coeff;



    do {

        token_prob = probs[vp8_coeff_band[i]][zero_nhood];



        if (!vp56_rac_get_prob_branchy(c, token_prob[0]))   // DCT_EOB

            return nonzero;



skip_eob:

        if (!vp56_rac_get_prob_branchy(c, token_prob[1])) { // DCT_0

            zero_nhood = 0;

            token_prob = probs[vp8_coeff_band[++i]][0];

            if (i < 16)

                goto skip_eob;

            return nonzero; // invalid input; blocks should end with EOB

        }



        if (!vp56_rac_get_prob_branchy(c, token_prob[2])) { // DCT_1

            coeff = 1;

            zero_nhood = 1;

        } else {

            zero_nhood = 2;



            if (!vp56_rac_get_prob_branchy(c, token_prob[3])) { // DCT 2,3,4

                coeff = vp56_rac_get_prob(c, token_prob[4]);

                if (coeff)

                    coeff += vp56_rac_get_prob(c, token_prob[5]);

                coeff += 2;

            } else {

                // DCT_CAT*

                if (!vp56_rac_get_prob_branchy(c, token_prob[6])) {

                    if (!vp56_rac_get_prob_branchy(c, token_prob[7])) { // DCT_CAT1

                        coeff  = 5 + vp56_rac_get_prob(c, vp8_dct_cat1_prob[0]);

                    } else {                                    // DCT_CAT2

                        coeff  = 7;

                        coeff += vp56_rac_get_prob(c, vp8_dct_cat2_prob[0]) << 1;

                        coeff += vp56_rac_get_prob(c, vp8_dct_cat2_prob[1]);

                    }

                } else {    // DCT_CAT3 and up

                    int a = vp56_rac_get_prob(c, token_prob[8]);

                    int b = vp56_rac_get_prob(c, token_prob[9+a]);

                    int cat = (a<<1) + b;

                    coeff  = 3 + (8<<cat);

                    coeff += vp8_rac_get_coeff(c, vp8_dct_cat_prob[cat]);

                }

            }

        }



        // todo: full [16] qmat? load into register?

        block[zigzag_scan[i]] = (vp8_rac_get(c) ? -coeff : coeff) * qmul[!!i];

        nonzero = ++i;

    } while (i < 16);



    return nonzero;

}
