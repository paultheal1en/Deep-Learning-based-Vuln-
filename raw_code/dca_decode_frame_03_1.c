static int dca_decode_frame(AVCodecContext * avctx,

                            void *data, int *data_size,

                            const uint8_t * buf, int buf_size)

{



    int i, j, k;

    int16_t *samples = data;

    DCAContext *s = avctx->priv_data;

    int channels;





    s->dca_buffer_size = dca_convert_bitstream(buf, buf_size, s->dca_buffer, DCA_MAX_FRAME_SIZE);

    if (s->dca_buffer_size == -1) {

        av_log(avctx, AV_LOG_ERROR, "Not a valid DCA frame\n");

        return -1;

    }



    init_get_bits(&s->gb, s->dca_buffer, s->dca_buffer_size * 8);

    if (dca_parse_frame_header(s) < 0) {

        //seems like the frame is corrupt, try with the next one

        *data_size=0;

        return buf_size;

    }

    //set AVCodec values with parsed data

    avctx->sample_rate = s->sample_rate;

    avctx->bit_rate = s->bit_rate;



    channels = s->prim_channels + !!s->lfe;

    if(avctx->request_channels == 2 && s->prim_channels > 2) {

        channels = 2;

        s->output = DCA_STEREO;

    }



    avctx->channels = channels;

    if(*data_size < (s->sample_blocks / 8) * 256 * sizeof(int16_t) * channels)

        return -1;

    *data_size = 0;

    for (i = 0; i < (s->sample_blocks / 8); i++) {

        dca_decode_block(s);

        s->dsp.float_to_int16(s->tsamples, s->samples, 256 * channels);

        /* interleave samples */

        for (j = 0; j < 256; j++) {

            for (k = 0; k < channels; k++)

                samples[k] = s->tsamples[j + k * 256];

            samples += channels;

        }

        *data_size += 256 * sizeof(int16_t) * channels;

    }



    return buf_size;

}
