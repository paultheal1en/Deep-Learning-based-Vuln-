static int decode_tns(AACContext * ac, TemporalNoiseShaping * tns,

        GetBitContext * gb, const IndividualChannelStream * ics) {

    int w, filt, i, coef_len, coef_res, coef_compress;

    const int is8 = ics->window_sequence[0] == EIGHT_SHORT_SEQUENCE;

    const int tns_max_order = is8 ? 7 : ac->m4ac.object_type == AOT_AAC_MAIN ? 20 : 12;

    for (w = 0; w < ics->num_windows; w++) {

        tns->n_filt[w] = get_bits(gb, 2 - is8);



        if (tns->n_filt[w])

            coef_res = get_bits1(gb);



        for (filt = 0; filt < tns->n_filt[w]; filt++) {

            int tmp2_idx;

            tns->length[w][filt] = get_bits(gb, 6 - 2*is8);



            if ((tns->order[w][filt] = get_bits(gb, 5 - 2*is8)) > tns_max_order) {

                av_log(ac->avccontext, AV_LOG_ERROR, "TNS filter order %d is greater than maximum %d.",

                       tns->order[w][filt], tns_max_order);

                tns->order[w][filt] = 0;

                return -1;

            }

            tns->direction[w][filt] = get_bits1(gb);

            coef_compress = get_bits1(gb);

            coef_len = coef_res + 3 - coef_compress;

            tmp2_idx = 2*coef_compress + coef_res;



            for (i = 0; i < tns->order[w][filt]; i++)

                tns->coef[w][filt][i] = tns_tmp2_map[tmp2_idx][get_bits(gb, coef_len)];

        }

    }

    return 0;

}
