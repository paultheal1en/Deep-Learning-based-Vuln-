int ff_parse_packing_format(int *ret, const char *arg, void *log_ctx)

{

    char *tail;

    int planar = strtol(arg, &tail, 10);

    if (*tail) {

        planar = (strcmp(arg, "packed") != 0);

    } else if (planar != 0 && planar != 1) {

        av_log(log_ctx, AV_LOG_ERROR, "Invalid packing format '%s'\n", arg);

        return AVERROR(EINVAL);

    }

    *ret = planar;

    return 0;

}
