static void ffserver_apply_stream_config(AVCodecContext *enc, const AVDictionary *conf, AVDictionary **opts)

{

    AVDictionaryEntry *e;



    /* Return values from ffserver_set_*_param are ignored.

       Values are initially parsed and checked before inserting to

       AVDictionary. */



    //video params

    if ((e = av_dict_get(conf, "VideoBitRateRangeMin", NULL, 0)))

        ffserver_set_int_param(&enc->rc_min_rate, e->value, 1000, INT_MIN,

                INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoBitRateRangeMax", NULL, 0)))

        ffserver_set_int_param(&enc->rc_max_rate, e->value, 1000, INT_MIN,

                INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "Debug", NULL, 0)))

        ffserver_set_int_param(&enc->debug, e->value, 0, INT_MIN, INT_MAX,

                NULL, 0, NULL);

    if ((e = av_dict_get(conf, "Strict", NULL, 0)))

        ffserver_set_int_param(&enc->strict_std_compliance, e->value, 0,

                INT_MIN, INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoBufferSize", NULL, 0)))

        ffserver_set_int_param(&enc->rc_buffer_size, e->value, 8*1024,

                INT_MIN, INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoBitRateTolerance", NULL, 0)))

        ffserver_set_int_param(&enc->bit_rate_tolerance, e->value, 1000,

                INT_MIN, INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoBitRate", NULL, 0)))

        ffserver_set_int_param(&enc->bit_rate, e->value, 1000, INT_MIN,

                INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoSizeWidth", NULL, 0)))

        ffserver_set_int_param(&enc->width, e->value, 0, INT_MIN, INT_MAX,

                NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoSizeHeight", NULL, 0)))

        ffserver_set_int_param(&enc->height, e->value, 0, INT_MIN, INT_MAX,

                NULL, 0, NULL);

    if ((e = av_dict_get(conf, "PixelFormat", NULL, 0))) {

        int val;

        ffserver_set_int_param(&val, e->value, 0, INT_MIN, INT_MAX, NULL, 0,

                NULL);

        enc->pix_fmt = val;

    }

    if ((e = av_dict_get(conf, "VideoGopSize", NULL, 0)))

        ffserver_set_int_param(&enc->gop_size, e->value, 0, INT_MIN, INT_MAX,

                NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoFrameRateNum", NULL, 0)))

        ffserver_set_int_param(&enc->time_base.num, e->value, 0, INT_MIN,

                INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoFrameRateDen", NULL, 0)))

        ffserver_set_int_param(&enc->time_base.den, e->value, 0, INT_MIN,

                INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoQDiff", NULL, 0)))

        ffserver_set_int_param(&enc->max_qdiff, e->value, 0, INT_MIN, INT_MAX,

                NULL, 0, NULL);

    if ((e = av_dict_get(conf, "VideoQMax", NULL, 0)))

        ffserver_set_int_param(&enc->qmax, e->value, 0, INT_MIN, INT_MAX, NULL,

                0, NULL);

    if ((e = av_dict_get(conf, "VideoQMin", NULL, 0)))

        ffserver_set_int_param(&enc->qmin, e->value, 0, INT_MIN, INT_MAX, NULL,

                0, NULL);

    if ((e = av_dict_get(conf, "LumiMask", NULL, 0)))

        ffserver_set_float_param(&enc->lumi_masking, e->value, 0, -FLT_MAX,

                FLT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "DarkMask", NULL, 0)))

        ffserver_set_float_param(&enc->dark_masking, e->value, 0, -FLT_MAX,

                FLT_MAX, NULL, 0, NULL);

    if (av_dict_get(conf, "BitExact", NULL, 0))

        enc->flags |= CODEC_FLAG_BITEXACT;

    if (av_dict_get(conf, "DctFastint", NULL, 0))

        enc->dct_algo  = FF_DCT_FASTINT;

    if (av_dict_get(conf, "IdctSimple", NULL, 0))

        enc->idct_algo = FF_IDCT_SIMPLE;

    if (av_dict_get(conf, "VideoHighQuality", NULL, 0))

        enc->mb_decision = FF_MB_DECISION_BITS;

    if ((e = av_dict_get(conf, "VideoTag", NULL, 0)))

        enc->codec_tag = MKTAG(e->value[0], e->value[1], e->value[2], e->value[3]);

    if (av_dict_get(conf, "Qscale", NULL, 0)) {

        enc->flags |= CODEC_FLAG_QSCALE;

        ffserver_set_int_param(&enc->global_quality, e->value, FF_QP2LAMBDA,

                INT_MIN, INT_MAX, NULL, 0, NULL);

    }

    if (av_dict_get(conf, "Video4MotionVector", NULL, 0)) {

        enc->mb_decision = FF_MB_DECISION_BITS; //FIXME remove

        enc->flags |= CODEC_FLAG_4MV;

    }

    //audio params

    if ((e = av_dict_get(conf, "AudioChannels", NULL, 0)))

        ffserver_set_int_param(&enc->channels, e->value, 0, INT_MIN, INT_MAX,

                NULL, 0, NULL);

    if ((e = av_dict_get(conf, "AudioSampleRate", NULL, 0)))

        ffserver_set_int_param(&enc->sample_rate, e->value, 0, INT_MIN,

                INT_MAX, NULL, 0, NULL);

    if ((e = av_dict_get(conf, "AudioBitRate", NULL, 0)))

        ffserver_set_int_param(&enc->bit_rate, e->value, 0, INT_MIN, INT_MAX,

                NULL, 0, NULL);



    av_opt_set_dict2(enc->priv_data, opts, AV_OPT_SEARCH_CHILDREN);

    av_opt_set_dict2(enc, opts, AV_OPT_SEARCH_CHILDREN);



    if (av_dict_count(*opts))

        av_log(NULL, AV_LOG_ERROR, "Something went wrong, %d options not set!!!\n", av_dict_count(*opts));

}
