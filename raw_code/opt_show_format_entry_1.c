static int opt_show_format_entry(void *optctx, const char *opt, const char *arg)
{
    char *buf = av_asprintf("format=%s", arg);
    int ret;
    av_log(NULL, AV_LOG_WARNING,
           "Option '%s' is deprecated, use '-show_entries format=%s' instead\n",
           opt, arg);
    ret = opt_show_entries(optctx, opt, buf);
    av_free(buf);
    return ret;
}