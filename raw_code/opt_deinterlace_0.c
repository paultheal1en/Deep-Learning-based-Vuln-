static int opt_deinterlace(void *optctx, const char *opt, const char *arg)

{

    av_log(NULL, AV_LOG_WARNING, "-%s is deprecated, use -filter:v yadif instead\n", opt);

    do_deinterlace = 1;

    return 0;

}
