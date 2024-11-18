static int parse_read_interval(const char *interval_spec,

                               ReadInterval *interval)

{

    int ret = 0;

    char *next, *p, *spec = av_strdup(interval_spec);

    if (!spec)

        return AVERROR(ENOMEM);



    if (!*spec) {

        av_log(NULL, AV_LOG_ERROR, "Invalid empty interval specification\n");

        ret = AVERROR(EINVAL);

        goto end;

    }



    p = spec;

    next = strchr(spec, '%');

    if (next)

        *next++ = 0;



    /* parse first part */

    if (*p) {

        interval->has_start = 1;



        if (*p == '+') {

            interval->start_is_offset = 1;

            p++;

        } else {

            interval->start_is_offset = 0;

        }



        ret = av_parse_time(&interval->start, p, 1);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR, "Invalid interval start specification '%s'\n", p);

            goto end;

        }

    } else {

        interval->has_start = 0;

    }



    /* parse second part */

    p = next;

    if (p && *p) {

        int64_t us;

        interval->has_end = 1;



        if (*p == '+') {

            interval->end_is_offset = 1;

            p++;

        } else {

            interval->end_is_offset = 0;

        }



        if (interval->end_is_offset && *p == '#') {

            long long int lli;

            char *tail;

            interval->duration_frames = 1;

            p++;

            lli = strtoll(p, &tail, 10);

            if (*tail || lli < 0) {

                av_log(NULL, AV_LOG_ERROR,

                       "Invalid or negative value '%s' for duration number of frames\n", p);

                goto end;

            }

            interval->end = lli;

        } else {


            ret = av_parse_time(&us, p, 1);

            if (ret < 0) {

                av_log(NULL, AV_LOG_ERROR, "Invalid interval end/duration specification '%s'\n", p);

                goto end;

            }

            interval->end = us;

        }

    } else {

        interval->has_end = 0;

    }



end:

    av_free(spec);

    return ret;

}