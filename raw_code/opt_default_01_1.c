int opt_default(const char *opt, const char *arg)

{

    const AVOption *oc, *of, *os, *oswr;

    char opt_stripped[128];

    const char *p;

    const AVClass *cc = avcodec_get_class(), *fc = avformat_get_class(), *sc, *swr_class;



    if (!(p = strchr(opt, ':')))

        p = opt + strlen(opt);

    av_strlcpy(opt_stripped, opt, FFMIN(sizeof(opt_stripped), p - opt + 1));



    if ((oc = av_opt_find(&cc, opt_stripped, NULL, 0,

                         AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)) ||

        ((opt[0] == 'v' || opt[0] == 'a' || opt[0] == 's') &&

         (oc = av_opt_find(&cc, opt + 1, NULL, 0, AV_OPT_SEARCH_FAKE_OBJ))))

        av_dict_set(&codec_opts, opt, arg, FLAGS(oc));

    if ((of = av_opt_find(&fc, opt, NULL, 0,

                          AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ)))

        av_dict_set(&format_opts, opt, arg, FLAGS(of));

#if CONFIG_SWSCALE

    sc = sws_get_class();

    if ((os = av_opt_find(&sc, opt, NULL, 0,

                          AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {

        // XXX we only support sws_flags, not arbitrary sws options

        int ret = av_opt_set(sws_opts, opt, arg, 0);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error setting option %s.\n", opt);

            return ret;

        }

    }

#endif

    swr_class = swr_get_class();

    if (!oc && !of && !os && (oswr = av_opt_find(&swr_class, opt, NULL, 0,

                          AV_OPT_SEARCH_CHILDREN | AV_OPT_SEARCH_FAKE_OBJ))) {

        int ret = av_opt_set(swr_opts, opt, arg, 0);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error setting option %s.\n", opt);

            return ret;

        }

    }



    if (oc || of || os || oswr)

        return 0;

    av_log(NULL, AV_LOG_ERROR, "Unrecognized option '%s'\n", opt);

    return AVERROR_OPTION_NOT_FOUND;

}
