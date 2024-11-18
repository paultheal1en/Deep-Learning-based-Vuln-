static int mp_decode_frame(MPADecodeContext *s, OUT_INT **samples,

                           const uint8_t *buf, int buf_size)

{

    int i, nb_frames, ch, ret;

    OUT_INT *samples_ptr;



    init_get_bits(&s->gb, buf + HEADER_SIZE, (buf_size - HEADER_SIZE) * 8);



    /* skip error protection field */

    if (s->error_protection)

        skip_bits(&s->gb, 16);



    switch(s->layer) {

    case 1:

        s->avctx->frame_size = 384;

        nb_frames = mp_decode_layer1(s);

        break;

    case 2:

        s->avctx->frame_size = 1152;

        nb_frames = mp_decode_layer2(s);

        break;

    case 3:

        s->avctx->frame_size = s->lsf ? 576 : 1152;

    default:

        nb_frames = mp_decode_layer3(s);



        if (nb_frames < 0)

            return nb_frames;



        s->last_buf_size=0;

        if (s->in_gb.buffer) {

            align_get_bits(&s->gb);

            i = get_bits_left(&s->gb)>>3;

            if (i >= 0 && i <= BACKSTEP_SIZE) {

                memmove(s->last_buf, s->gb.buffer + (get_bits_count(&s->gb)>>3), i);

                s->last_buf_size=i;

            } else

                av_log(s->avctx, AV_LOG_ERROR, "invalid old backstep %d\n", i);

            s->gb           = s->in_gb;

            s->in_gb.buffer = NULL;

        }



        align_get_bits(&s->gb);

        assert((get_bits_count(&s->gb) & 7) == 0);

        i = get_bits_left(&s->gb) >> 3;



        if (i < 0 || i > BACKSTEP_SIZE || nb_frames < 0) {

            if (i < 0)

                av_log(s->avctx, AV_LOG_ERROR, "invalid new backstep %d\n", i);

            i = FFMIN(BACKSTEP_SIZE, buf_size - HEADER_SIZE);

        }

        assert(i <= buf_size - HEADER_SIZE && i >= 0);

        memcpy(s->last_buf + s->last_buf_size, s->gb.buffer + buf_size - HEADER_SIZE - i, i);

        s->last_buf_size += i;

    }



    /* get output buffer */

    if (!samples) {

        av_assert0(s->frame != NULL);

        s->frame->nb_samples = s->avctx->frame_size;

        if ((ret = ff_get_buffer(s->avctx, s->frame, 0)) < 0) {

            av_log(s->avctx, AV_LOG_ERROR, "get_buffer() failed\n");

            return ret;

        }

        samples = (OUT_INT **)s->frame->extended_data;

    }



    /* apply the synthesis filter */

    for (ch = 0; ch < s->nb_channels; ch++) {

        int sample_stride;

        if (s->avctx->sample_fmt == OUT_FMT_P) {

            samples_ptr   = samples[ch];

            sample_stride = 1;

        } else {

            samples_ptr   = samples[0] + ch;

            sample_stride = s->nb_channels;

        }

        for (i = 0; i < nb_frames; i++) {

            RENAME(ff_mpa_synth_filter)(&s->mpadsp, s->synth_buf[ch],

                                        &(s->synth_buf_offset[ch]),

                                        RENAME(ff_mpa_synth_window),

                                        &s->dither_state, samples_ptr,

                                        sample_stride, s->sb_samples[ch][i]);

            samples_ptr += 32 * sample_stride;

        }

    }



    return nb_frames * 32 * sizeof(OUT_INT) * s->nb_channels;

}
