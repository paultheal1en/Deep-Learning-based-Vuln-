int ff_intrax8_decode_picture(IntraX8Context *const w, int dquant,

                              int quant_offset)

{

    MpegEncContext *const s = w->s;

    int mb_xy;

    assert(s);

    w->use_quant_matrix = get_bits1(&s->gb);



    w->dquant = dquant;

    w->quant  = dquant >> 1;

    w->qsum   = quant_offset;



    w->divide_quant_dc_luma = ((1 << 16) + (w->quant >> 1)) / w->quant;

    if (w->quant < 5) {

        w->quant_dc_chroma        = w->quant;

        w->divide_quant_dc_chroma = w->divide_quant_dc_luma;

    } else {

        w->quant_dc_chroma        = w->quant + ((w->quant + 3) >> 3);

        w->divide_quant_dc_chroma = ((1 << 16) + (w->quant_dc_chroma >> 1)) / w->quant_dc_chroma;

    }

    x8_reset_vlc_tables(w);



    for (s->mb_y = 0; s->mb_y < s->mb_height * 2; s->mb_y++) {

        x8_init_block_index(w, s->current_picture.f, s->mb_y);

        mb_xy = (s->mb_y >> 1) * s->mb_stride;



        for (s->mb_x = 0; s->mb_x < s->mb_width * 2; s->mb_x++) {

            x8_get_prediction(w);

            if (x8_setup_spatial_predictor(w, 0))

                goto error;

            if (x8_decode_intra_mb(w, 0))

                goto error;



            if (s->mb_x & s->mb_y & 1) {

                x8_get_prediction_chroma(w);



                /* when setting up chroma, no vlc is read,

                 * so no error condition can be reached */

                x8_setup_spatial_predictor(w, 1);

                if (x8_decode_intra_mb(w, 1))

                    goto error;



                x8_setup_spatial_predictor(w, 2);

                if (x8_decode_intra_mb(w, 2))

                    goto error;



                w->dest[1] += 8;

                w->dest[2] += 8;



                /* emulate MB info in the relevant tables */

                s->mbskip_table[mb_xy]                 = 0;

                s->mbintra_table[mb_xy]                = 1;

                s->current_picture.qscale_table[mb_xy] = w->quant;

                mb_xy++;

            }

            w->dest[0] += 8;

        }

        if (s->mb_y & 1)

            ff_mpeg_draw_horiz_band(s, (s->mb_y - 1) * 8, 16);

    }



error:

    return 0;

}
