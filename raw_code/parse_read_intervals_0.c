static int parse_read_intervals(const char *intervals_spec)

{

    int ret, n, i;

    char *p, *spec = av_strdup(intervals_spec);

    if (!spec)

        return AVERROR(ENOMEM);



    /* preparse specification, get number of intervals */

    for (n = 0, p = spec; *p; p++)

        if (*p == ',')

            n++;

    n++;



    read_intervals = av_malloc(n * sizeof(*read_intervals));

    if (!read_intervals) {

        ret = AVERROR(ENOMEM);

        goto end;

    }

    read_intervals_nb = n;



    /* parse intervals */

    p = spec;

    for (i = 0; i < n; i++) {

        char *next = strchr(p, ',');

        if (next)

            *next++ = 0;



        read_intervals[i].id = i;

        ret = parse_read_interval(p, &read_intervals[i]);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Error parsing read interval #%d '%s'\n",

                   i, p);

            goto end;

        }

        av_log(NULL, AV_LOG_VERBOSE, "Parsed log interval ");

        log_read_interval(&read_intervals[i], NULL, AV_LOG_VERBOSE);

        p = next;

        av_assert0(i <= read_intervals_nb);

    }

    av_assert0(i == read_intervals_nb);



end:

    av_free(spec);

    return ret;

}
