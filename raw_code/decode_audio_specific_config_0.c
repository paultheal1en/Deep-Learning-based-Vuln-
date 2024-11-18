static int decode_audio_specific_config(AACContext *ac,

                                        AVCodecContext *avctx,

                                        MPEG4AudioConfig *m4ac,

                                        const uint8_t *data, int bit_size,

                                        int sync_extension)

{

    GetBitContext gb;

    int i, ret;



    av_dlog(avctx, "extradata size %d\n", avctx->extradata_size);

    for (i = 0; i < avctx->extradata_size; i++)

        av_dlog(avctx, "%02x ", avctx->extradata[i]);

    av_dlog(avctx, "\n");



    init_get_bits(&gb, data, bit_size);



    if ((i = avpriv_mpeg4audio_get_config(m4ac, data, bit_size,

                                          sync_extension)) < 0)

        return AVERROR_INVALIDDATA;

    if (m4ac->sampling_index > 12) {

        av_log(avctx, AV_LOG_ERROR,

               "invalid sampling rate index %d\n",

               m4ac->sampling_index);

        return AVERROR_INVALIDDATA;

    }



    skip_bits_long(&gb, i);



    switch (m4ac->object_type) {

    case AOT_AAC_MAIN:

    case AOT_AAC_LC:

    case AOT_AAC_LTP:

        if ((ret = decode_ga_specific_config(ac, avctx, &gb,

                                            m4ac, m4ac->chan_config)) < 0)

            return ret;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR,

               "Audio object type %s%d is not supported.\n",

               m4ac->sbr == 1 ? "SBR+" : "",

               m4ac->object_type);

        return AVERROR(ENOSYS);

    }



    av_dlog(avctx,

            "AOT %d chan config %d sampling index %d (%d) SBR %d PS %d\n",

            m4ac->object_type, m4ac->chan_config, m4ac->sampling_index,

            m4ac->sample_rate, m4ac->sbr,

            m4ac->ps);



    return get_bits_count(&gb);

}
