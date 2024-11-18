static av_cold int alac_encode_init(AVCodecContext *avctx)

{

    AlacEncodeContext *s = avctx->priv_data;

    int ret;

    uint8_t *alac_extradata;



    avctx->frame_size = s->frame_size = DEFAULT_FRAME_SIZE;



    if (avctx->sample_fmt != AV_SAMPLE_FMT_S16) {

        av_log(avctx, AV_LOG_ERROR, "only pcm_s16 input samples are supported\n");

        return -1;

    }



    /* TODO: Correctly implement multi-channel ALAC.

             It is similar to multi-channel AAC, in that it has a series of

             single-channel (SCE), channel-pair (CPE), and LFE elements. */

    if (avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "only mono or stereo input is currently supported\n");

        return AVERROR_PATCHWELCOME;

    }



    // Set default compression level

    if (avctx->compression_level == FF_COMPRESSION_DEFAULT)

        s->compression_level = 2;

    else

        s->compression_level = av_clip(avctx->compression_level, 0, 2);



    // Initialize default Rice parameters

    s->rc.history_mult    = 40;

    s->rc.initial_history = 10;

    s->rc.k_modifier      = 14;

    s->rc.rice_modifier   = 4;



    s->max_coded_frame_size = get_max_frame_size(avctx->frame_size,

                                                 avctx->channels,

                                                 DEFAULT_SAMPLE_SIZE);



    // FIXME: consider wasted_bytes

    s->write_sample_size  = DEFAULT_SAMPLE_SIZE + avctx->channels - 1;



    avctx->extradata = av_mallocz(ALAC_EXTRADATA_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!avctx->extradata) {

        ret = AVERROR(ENOMEM);

        goto error;

    }

    avctx->extradata_size = ALAC_EXTRADATA_SIZE;



    alac_extradata = avctx->extradata;

    AV_WB32(alac_extradata,    ALAC_EXTRADATA_SIZE);

    AV_WB32(alac_extradata+4,  MKBETAG('a','l','a','c'));

    AV_WB32(alac_extradata+12, avctx->frame_size);

    AV_WB8 (alac_extradata+17, DEFAULT_SAMPLE_SIZE);

    AV_WB8 (alac_extradata+21, avctx->channels);

    AV_WB32(alac_extradata+24, s->max_coded_frame_size);

    AV_WB32(alac_extradata+28,

            avctx->sample_rate * avctx->channels * DEFAULT_SAMPLE_SIZE); // average bitrate

    AV_WB32(alac_extradata+32, avctx->sample_rate);



    // Set relevant extradata fields

    if (s->compression_level > 0) {

        AV_WB8(alac_extradata+18, s->rc.history_mult);

        AV_WB8(alac_extradata+19, s->rc.initial_history);

        AV_WB8(alac_extradata+20, s->rc.k_modifier);

    }



    s->min_prediction_order = DEFAULT_MIN_PRED_ORDER;

    if (avctx->min_prediction_order >= 0) {

        if (avctx->min_prediction_order < MIN_LPC_ORDER ||

           avctx->min_prediction_order > ALAC_MAX_LPC_ORDER) {

            av_log(avctx, AV_LOG_ERROR, "invalid min prediction order: %d\n",

                   avctx->min_prediction_order);

            ret = AVERROR(EINVAL);

            goto error;

        }



        s->min_prediction_order = avctx->min_prediction_order;

    }



    s->max_prediction_order = DEFAULT_MAX_PRED_ORDER;

    if (avctx->max_prediction_order >= 0) {

        if (avctx->max_prediction_order < MIN_LPC_ORDER ||

            avctx->max_prediction_order > ALAC_MAX_LPC_ORDER) {

            av_log(avctx, AV_LOG_ERROR, "invalid max prediction order: %d\n",

                   avctx->max_prediction_order);

            ret = AVERROR(EINVAL);

            goto error;

        }



        s->max_prediction_order = avctx->max_prediction_order;

    }



    if (s->max_prediction_order < s->min_prediction_order) {

        av_log(avctx, AV_LOG_ERROR,

               "invalid prediction orders: min=%d max=%d\n",

               s->min_prediction_order, s->max_prediction_order);

        ret = AVERROR(EINVAL);

        goto error;

    }



    avctx->coded_frame = avcodec_alloc_frame();

    if (!avctx->coded_frame) {

        ret = AVERROR(ENOMEM);

        goto error;

    }



    s->avctx = avctx;



    if ((ret = ff_lpc_init(&s->lpc_ctx, avctx->frame_size,

                           s->max_prediction_order,

                           FF_LPC_TYPE_LEVINSON)) < 0) {

        goto error;

    }



    return 0;

error:

    alac_encode_close(avctx);

    return ret;

}
