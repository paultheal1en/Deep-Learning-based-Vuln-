void avpriv_tak_parse_streaminfo(GetBitContext *gb, TAKStreamInfo *s)

{

    uint64_t channel_mask = 0;

    int frame_type, i;



    s->codec = get_bits(gb, TAK_ENCODER_CODEC_BITS);

    skip_bits(gb, TAK_ENCODER_PROFILE_BITS);



    frame_type = get_bits(gb, TAK_SIZE_FRAME_DURATION_BITS);

    s->samples = get_bits64(gb, TAK_SIZE_SAMPLES_NUM_BITS);



    s->data_type   = get_bits(gb, TAK_FORMAT_DATA_TYPE_BITS);

    s->sample_rate = get_bits(gb, TAK_FORMAT_SAMPLE_RATE_BITS) +

                     TAK_SAMPLE_RATE_MIN;

    s->bps         = get_bits(gb, TAK_FORMAT_BPS_BITS) +

                     TAK_BPS_MIN;

    s->channels    = get_bits(gb, TAK_FORMAT_CHANNEL_BITS) +

                     TAK_CHANNELS_MIN;



    if (get_bits1(gb)) {

        skip_bits(gb, TAK_FORMAT_VALID_BITS);

        if (get_bits1(gb)) {

            for (i = 0; i < s->channels; i++) {

                int value = get_bits(gb, TAK_FORMAT_CH_LAYOUT_BITS);



                if (value < FF_ARRAY_ELEMS(tak_channel_layouts))

                    channel_mask |= tak_channel_layouts[value];

            }

        }

    }



    s->ch_layout     = channel_mask;

    s->frame_samples = tak_get_nb_samples(s->sample_rate, frame_type);

}
