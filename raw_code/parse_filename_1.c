static int parse_filename(char *filename, char **representation_id,

                          char **initialization_pattern, char **media_pattern) {

    char *underscore_pos = NULL;

    char *period_pos = NULL;

    char *temp_pos = NULL;

    char *filename_str = av_strdup(filename);

    if (!filename_str) return AVERROR(ENOMEM);

    temp_pos = av_stristr(filename_str, "_");

    while (temp_pos) {

        underscore_pos = temp_pos + 1;

        temp_pos = av_stristr(temp_pos + 1, "_");

    }

    if (!underscore_pos) return -1;

    period_pos = av_stristr(underscore_pos, ".");

    if (!period_pos) return -1;

    *(underscore_pos - 1) = 0;

    if (representation_id) {

        *representation_id = av_malloc(period_pos - underscore_pos + 1);

        if (!(*representation_id)) return AVERROR(ENOMEM);

        av_strlcpy(*representation_id, underscore_pos, period_pos - underscore_pos + 1);

    }

    if (initialization_pattern) {

        *initialization_pattern = av_asprintf("%s_$RepresentationID$.hdr",

                                              filename_str);

        if (!(*initialization_pattern)) return AVERROR(ENOMEM);

    }

    if (media_pattern) {

        *media_pattern = av_asprintf("%s_$RepresentationID$_$Number$.chk",

                                     filename_str);

        if (!(*media_pattern)) return AVERROR(ENOMEM);

    }

    av_free(filename_str);

    return 0;

}
