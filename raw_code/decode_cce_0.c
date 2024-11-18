static int decode_cce(AACContext * ac, GetBitContext * gb, ChannelElement * che) {

    int num_gain = 0;

    int c, g, sfb, ret, idx = 0;

    int sign;

    float scale;

    SingleChannelElement * sce = &che->ch[0];

    ChannelCoupling * coup     = &che->coup;



    coup->coupling_point = 2*get_bits1(gb);

    coup->num_coupled = get_bits(gb, 3);

    for (c = 0; c <= coup->num_coupled; c++) {

        num_gain++;

        coup->type[c] = get_bits1(gb) ? TYPE_CPE : TYPE_SCE;

        coup->id_select[c] = get_bits(gb, 4);

        if (coup->type[c] == TYPE_CPE) {

            coup->ch_select[c] = get_bits(gb, 2);

            if (coup->ch_select[c] == 3)

                num_gain++;

        } else

            coup->ch_select[c] = 1;

    }

    coup->coupling_point += get_bits1(gb);



    if (coup->coupling_point == 2) {

        av_log(ac->avccontext, AV_LOG_ERROR,

            "Independently switched CCE with 'invalid' domain signalled.\n");

        memset(coup, 0, sizeof(ChannelCoupling));

        return -1;

    }



    sign = get_bits(gb, 1);

    scale = pow(2., pow(2., (int)get_bits(gb, 2) - 3));



    if ((ret = decode_ics(ac, sce, gb, 0, 0)))

        return ret;



    for (c = 0; c < num_gain; c++) {

        int cge = 1;

        int gain = 0;

        float gain_cache = 1.;

        if (c) {

            cge = coup->coupling_point == AFTER_IMDCT ? 1 : get_bits1(gb);

            gain = cge ? get_vlc2(gb, vlc_scalefactors.table, 7, 3) - 60: 0;

            gain_cache = pow(scale, gain);

        }

        for (g = 0; g < sce->ics.num_window_groups; g++) {

            for (sfb = 0; sfb < sce->ics.max_sfb; sfb++, idx++) {

                if (sce->band_type[idx] != ZERO_BT) {

                    if (!cge) {

                        int t = get_vlc2(gb, vlc_scalefactors.table, 7, 3) - 60;

                        if (t) {

                            int s = 1;

                            if (sign) {

                                s  -= 2 * (t & 0x1);

                                t >>= 1;

                            }

                            gain += t;

                            gain_cache = pow(scale, gain) * s;

                        }

                    }

                    coup->gain[c][idx] = gain_cache;

                }

            }

        }

    }

    return 0;

}
