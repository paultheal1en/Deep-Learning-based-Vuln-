static int ffserver_set_int_param(int *dest, const char *value, int factor,

                                  int min, int max, FFServerConfig *config,

                                  const char *error_msg, ...)

{

    int tmp;

    char *tailp;

    if (!value || !value[0])

        goto error;

    errno = 0;

    tmp = strtol(value, &tailp, 0);

    if (tmp < min || tmp > max)

        goto error;

    if (factor) {

        if (FFABS(tmp) > INT_MAX / FFABS(factor))

            goto error;

        tmp *= factor;

    }

    if (tailp[0] || errno)

        goto error;

    if (dest)

        *dest = tmp;

    return 0;

  error:

    if (config) {

        va_list vl;

        va_start(vl, error_msg);

        vreport_config_error(config->filename, config->line_num, AV_LOG_ERROR,

                &config->errors, error_msg, vl);

        va_end(vl);

    }

    return AVERROR(EINVAL);

}
