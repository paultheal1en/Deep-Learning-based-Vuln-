static int aac_decode_er_frame(AVCodecContext *avctx, void *data,

                               int *got_frame_ptr, GetBitContext *gb)

{

    AACContext *ac = avctx->priv_data;

    const MPEG4AudioConfig *const m4ac = &ac->oc[1].m4ac;

    ChannelElement *che;

    int err, i;

    int samples = m4ac->frame_length_short ? 960 : 1024;

    int chan_config = m4ac->chan_config;

    int aot = m4ac->object_type;



    if (aot == AOT_ER_AAC_LD || aot == AOT_ER_AAC_ELD)

        samples >>= 1;



    ac->frame = data;



    if ((err = frame_configure_elements(avctx)) < 0)

        return err;



    // The FF_PROFILE_AAC_* defines are all object_type - 1

    // This may lead to an undefined profile being signaled

    ac->avctx->profile = aot - 1;



    ac->tags_mapped = 0;



    if (chan_config < 0 || (chan_config >= 8 && chan_config < 11) || chan_config >= 13) {

        avpriv_request_sample(avctx, "Unknown ER channel configuration %d",

                              chan_config);

        return AVERROR_INVALIDDATA;

    }

    for (i = 0; i < tags_per_config[chan_config]; i++) {

        const int elem_type = aac_channel_layout_map[chan_config-1][i][0];

        const int elem_id   = aac_channel_layout_map[chan_config-1][i][1];

        if (!(che=get_che(ac, elem_type, elem_id))) {

            av_log(ac->avctx, AV_LOG_ERROR,

                   "channel element %d.%d is not allocated\n",

                   elem_type, elem_id);

            return AVERROR_INVALIDDATA;

        }

        che->present = 1;

        if (aot != AOT_ER_AAC_ELD)

            skip_bits(gb, 4);

        switch (elem_type) {

        case TYPE_SCE:

            err = decode_ics(ac, &che->ch[0], gb, 0, 0);

            break;

        case TYPE_CPE:

            err = decode_cpe(ac, gb, che);

            break;

        case TYPE_LFE:

            err = decode_ics(ac, &che->ch[0], gb, 0, 0);

            break;

        }

        if (err < 0)

            return err;

    }



    spectral_to_sample(ac);



    ac->frame->nb_samples = samples;

    ac->frame->sample_rate = avctx->sample_rate;

    *got_frame_ptr = 1;



    skip_bits_long(gb, get_bits_left(gb));

    return 0;

}
