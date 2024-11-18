static void ac3_decode_transform_coeffs_ch(AC3DecodeContext *s, int ch_index, mant_groups *m)

{

    int start_freq = s->start_freq[ch_index];

    int end_freq   = s->end_freq[ch_index];

    uint8_t *baps  = s->bap[ch_index];

    int8_t *exps   = s->dexps[ch_index];

    int32_t *coeffs = s->fixed_coeffs[ch_index];

    int dither     = (ch_index == CPL_CH) || s->dither_flag[ch_index];

    GetBitContext *gbc = &s->gbc;

    int freq;



    for (freq = start_freq; freq < end_freq; freq++) {

        int bap = baps[freq];

        int mantissa;

        switch (bap) {

        case 0:

            /* random noise with approximate range of -0.707 to 0.707 */

            if (dither)

                mantissa = (((av_lfg_get(&s->dith_state)>>8)*181)>>8) - 5931008;

            else

                mantissa = 0;

            break;

        case 1:

            if (m->b1) {

                m->b1--;

                mantissa = m->b1_mant[m->b1];

            } else {

                int bits      = get_bits(gbc, 5);

                mantissa      = b1_mantissas[bits][0];

                m->b1_mant[1] = b1_mantissas[bits][1];

                m->b1_mant[0] = b1_mantissas[bits][2];

                m->b1         = 2;


            break;

        case 2:

            if (m->b2) {

                m->b2--;

                mantissa = m->b2_mant[m->b2];

            } else {

                int bits      = get_bits(gbc, 7);

                mantissa      = b2_mantissas[bits][0];

                m->b2_mant[1] = b2_mantissas[bits][1];

                m->b2_mant[0] = b2_mantissas[bits][2];

                m->b2         = 2;


            break;

        case 3:

            mantissa = b3_mantissas[get_bits(gbc, 3)];

            break;

        case 4:

            if (m->b4) {

                m->b4 = 0;

                mantissa = m->b4_mant;

            } else {

                int bits   = get_bits(gbc, 7);

                mantissa   = b4_mantissas[bits][0];

                m->b4_mant = b4_mantissas[bits][1];

                m->b4      = 1;


            break;

        case 5:

            mantissa = b5_mantissas[get_bits(gbc, 4)];

            break;

        default: /* 6 to 15 */

            /* Shift mantissa and sign-extend it. */





            mantissa = get_sbits(gbc, quantization_tab[bap]);

            mantissa <<= 24 - quantization_tab[bap];

            break;


        coeffs[freq] = mantissa >> exps[freq];

