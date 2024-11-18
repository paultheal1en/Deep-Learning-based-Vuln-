static av_cold int alac_decode_init(AVCodecContext * avctx)

{

    int ret;

    int req_packed;

    ALACContext *alac = avctx->priv_data;

    alac->avctx = avctx;



    /* initialize from the extradata */

    if (alac->avctx->extradata_size != ALAC_EXTRADATA_SIZE) {

        av_log(avctx, AV_LOG_ERROR, "expected %d extradata bytes\n",

            ALAC_EXTRADATA_SIZE);

        return -1;

    }

    if (alac_set_info(alac)) {

        av_log(avctx, AV_LOG_ERROR, "set_info failed\n");

        return -1;

    }



    req_packed = LIBAVCODEC_VERSION_MAJOR < 55 && !av_sample_fmt_is_planar(avctx->request_sample_fmt);

    switch (alac->sample_size) {

    case 16: avctx->sample_fmt = req_packed ? AV_SAMPLE_FMT_S16 : AV_SAMPLE_FMT_S16P;

             break;

    case 24:

    case 32: avctx->sample_fmt = req_packed ? AV_SAMPLE_FMT_S32 : AV_SAMPLE_FMT_S32P;

             break;

    default: av_log_ask_for_sample(avctx, "Sample depth %d is not supported.\n",

                                   alac->sample_size);

             return AVERROR_PATCHWELCOME;

    }

    avctx->bits_per_raw_sample = alac->sample_size;



    if (alac->channels < 1) {

        av_log(avctx, AV_LOG_WARNING, "Invalid channel count\n");

        alac->channels = avctx->channels;

    } else {

        if (alac->channels > MAX_CHANNELS)

            alac->channels = avctx->channels;

        else

            avctx->channels = alac->channels;

    }

    if (avctx->channels > MAX_CHANNELS) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported channel count: %d\n",

               avctx->channels);

        return AVERROR_PATCHWELCOME;

    }

    avctx->channel_layout = alac_channel_layouts[alac->channels - 1];



    if ((ret = allocate_buffers(alac)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "Error allocating buffers\n");

        return ret;

    }



    avcodec_get_frame_defaults(&alac->frame);

    avctx->coded_frame = &alac->frame;



    return 0;

}
