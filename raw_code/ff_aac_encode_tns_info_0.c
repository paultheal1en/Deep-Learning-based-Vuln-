void ff_aac_encode_tns_info(AACEncContext *s, SingleChannelElement *sce)

{

    int i, w, filt, coef_len, coef_compress;

    const int coef_res = MAX_LPC_PRECISION == 4 ? 1 : 0;

    const int is8 = sce->ics.window_sequence[0] == EIGHT_SHORT_SEQUENCE;



    put_bits(&s->pb, 1, !!sce->tns.present);



    if (!sce->tns.present)

        return;



    for (i = 0; i < sce->ics.num_windows; i++) {

        put_bits(&s->pb, 2 - is8, sce->tns.n_filt[i]);

        if (sce->tns.n_filt[i]) {

            put_bits(&s->pb, 1, !!coef_res);

            for (filt = 0; filt < sce->tns.n_filt[i]; filt++) {

                put_bits(&s->pb, 6 - 2 * is8, sce->tns.length[i][filt]);

                put_bits(&s->pb, 5 - 2 * is8, sce->tns.order[i][filt]);

                if (sce->tns.order[i][filt]) {

                    coef_compress = compress_coef(sce->tns.coef_idx[i][filt],

                                                  sce->tns.order[i][filt]);

                    put_bits(&s->pb, 1, !!sce->tns.direction[i][filt]);

                    put_bits(&s->pb, 1, !!coef_compress);

                    coef_len = coef_res + 3 - coef_compress;

                    for (w = 0; w < sce->tns.order[i][filt]; w++)

                        put_bits(&s->pb, coef_len, sce->tns.coef_idx[i][filt][w]);

                }

            }

        }

    }

}
