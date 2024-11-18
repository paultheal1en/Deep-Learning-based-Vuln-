static AVFilterContext *create_filter_with_args(const char *filt, void *opaque)
{
    AVFilterContext *ret;
    char *filter = av_strdup(filt); /* copy - don't mangle the input string */
    char *name, *args;
    name = filter;
    if((args = strchr(filter, '='))) {
        /* ensure we at least have a name */
        if(args == filter)
            goto fail;
        *args ++ = 0;
    }
    av_log(NULL, AV_LOG_INFO, "creating filter \"%s\" with args \"%s\"\n",
           name, args ? args : "(none)");
    if((ret = avfilter_create_by_name(name, NULL))) {
        if(avfilter_init_filter(ret, args, opaque)) {
            av_log(NULL, AV_LOG_ERROR, "error initializing filter!\n");
            avfilter_destroy(ret);
            goto fail;
        }
    } else av_log(NULL, AV_LOG_ERROR, "error creating filter!\n");
    return ret;
fail:
    return NULL;
}