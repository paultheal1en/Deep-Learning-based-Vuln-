static int set_string_fmt(void *obj, const AVOption *o, const char *val, uint8_t *dst,

                          int fmt_nb, int ((*get_fmt)(const char *)), const char *desc)

{

    int fmt;



    if (!val || !strcmp(val, "none")) {

        fmt = -1;

    } else {

        fmt = get_fmt(val);

        if (fmt == -1) {

            char *tail;

            fmt = strtol(val, &tail, 0);

            if (*tail || (unsigned)fmt >= fmt_nb) {

                av_log(obj, AV_LOG_ERROR,

                       "Unable to parse option value \"%s\" as %s\n", val, desc);

                return AVERROR(EINVAL);

            }

        }

    }



    *(int *)dst = fmt;

    return 0;

}
