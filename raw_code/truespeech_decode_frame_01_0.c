static int truespeech_decode_frame(AVCodecContext *avctx,

                void *data, int *data_size,

                uint8_t *buf, int buf_size)

{

    TSContext *c = avctx->priv_data;



    int i;

    short *samples = data;

    int consumed = 0;

    int16_t out_buf[240];



    if (!buf_size)

        return 0;



    while (consumed < buf_size) {

        truespeech_read_frame(c, buf + consumed);

        consumed += 32;



        truespeech_correlate_filter(c);

        truespeech_filters_merge(c);



        memset(out_buf, 0, 240 * 2);

        for(i = 0; i < 4; i++) {

            truespeech_apply_twopoint_filter(c, i);

            truespeech_place_pulses(c, out_buf + i * 60, i);

            truespeech_update_filters(c, out_buf + i * 60, i);

            truespeech_synth(c, out_buf + i * 60, i);

        }



        truespeech_save_prevvec(c);



        /* finally output decoded frame */

        for(i = 0; i < 240; i++)

            *samples++ = out_buf[i];



    }



    *data_size = consumed * 15;



    return buf_size;

}
