int opt_codec_debug(void *optctx, const char *opt, const char *arg)

{

    av_log_set_level(AV_LOG_DEBUG);

    return opt_default(NULL, opt, arg);

}
