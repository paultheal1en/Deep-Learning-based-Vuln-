static int decode_frame(AVCodecContext *avctx, void *data,

                        int *got_frame_ptr, AVPacket *pkt)

{

    GetBitContext  gb;

    VimaContext    *vima = avctx->priv_data;

    int16_t        pcm_data[2];

    uint32_t       samples;

    int8_t         channel_hint[2];

    int            ret, chan, channels = 1;



    init_get_bits(&gb, pkt->data, pkt->size * 8);



    if (pkt->size < 13)

        return AVERROR_INVALIDDATA;



    samples = get_bits_long(&gb, 32);

    if (samples == 0xffffffff) {

        skip_bits_long(&gb, 32);

        samples = get_bits_long(&gb, 32);

    }



    if (samples > pkt->size * 2)

        return AVERROR_INVALIDDATA;



    channel_hint[0] = get_sbits(&gb, 8);

    if (channel_hint[0] & 0x80) {

        channel_hint[0] = ~channel_hint[0];

        channels = 2;

    }

    avctx->channels = channels;

    avctx->channel_layout = (channels == 2) ? AV_CH_LAYOUT_STEREO :

                                              AV_CH_LAYOUT_MONO;

    pcm_data[0] = get_sbits(&gb, 16);

    if (channels > 1) {

        channel_hint[1] = get_sbits(&gb, 8);

        pcm_data[1] = get_sbits(&gb, 16);

    }



    vima->frame.nb_samples = samples;

    if ((ret = avctx->get_buffer(avctx, &vima->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }



    for (chan = 0; chan < channels; chan++) {

        uint16_t *dest = (uint16_t*)vima->frame.data[0] + chan;

        int step_index = channel_hint[chan];

        int output = pcm_data[chan];

        int sample;



        for (sample = 0; sample < samples; sample++) {

            int lookup_size, lookup, highbit, lowbits;



            step_index  = av_clip(step_index, 0, 88);

            lookup_size = size_table[step_index];

            lookup      = get_bits(&gb, lookup_size);

            highbit     = 1 << (lookup_size - 1);

            lowbits     = highbit - 1;



            if (lookup & highbit)

                lookup ^= highbit;

            else

                highbit = 0;



            if (lookup == lowbits) {

                output = get_sbits(&gb, 16);

            } else {

                int predict_index, diff;



                predict_index = (lookup << (7 - lookup_size)) | (step_index << 6);

                predict_index = av_clip(predict_index, 0, 5785);

                diff          = vima->predict_table[predict_index];

                if (lookup)

                    diff += ff_adpcm_step_table[step_index] >> (lookup_size - 1);

                if (highbit)

                    diff  = -diff;



                output  = av_clip_int16(output + diff);

            }



            *dest = output;

            dest += channels;



            step_index += step_index_tables[lookup_size - 2][lookup];

        }

    }



    *got_frame_ptr   = 1;

    *(AVFrame *)data = vima->frame;



    return pkt->size;

}
