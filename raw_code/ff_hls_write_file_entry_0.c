int ff_hls_write_file_entry(AVIOContext *out, int insert_discont,

                             int byterange_mode,

                             double duration, int round_duration,

                             int64_t size, int64_t pos, //Used only if HLS_SINGLE_FILE flag is set

                             char *baseurl, //Ignored if NULL

                             char *filename, double *prog_date_time) {

    if (!out || !filename)

        return AVERROR(EINVAL);



    if (insert_discont) {

        avio_printf(out, "#EXT-X-DISCONTINUITY\n");

    }

    if (round_duration)

        avio_printf(out, "#EXTINF:%ld,\n",  lrint(duration));

    else

        avio_printf(out, "#EXTINF:%f,\n", duration);

    if (byterange_mode)

        avio_printf(out, "#EXT-X-BYTERANGE:%"PRId64"@%"PRId64"\n", size, pos);



    if (prog_date_time) {

        time_t tt, wrongsecs;

        int milli;

        struct tm *tm, tmpbuf;

        char buf0[128], buf1[128];

        tt = (int64_t)*prog_date_time;

        milli = av_clip(lrint(1000*(*prog_date_time - tt)), 0, 999);

        tm = localtime_r(&tt, &tmpbuf);

        strftime(buf0, sizeof(buf0), "%Y-%m-%dT%H:%M:%S", tm);

        if (!strftime(buf1, sizeof(buf1), "%z", tm) || buf1[1]<'0' ||buf1[1]>'2') {

            int tz_min, dst = tm->tm_isdst;

            tm = gmtime_r(&tt, &tmpbuf);

            tm->tm_isdst = dst;

            wrongsecs = mktime(tm);

            tz_min = (FFABS(wrongsecs - tt) + 30) / 60;

            snprintf(buf1, sizeof(buf1),

                     "%c%02d%02d",

                     wrongsecs <= tt ? '+' : '-',

                     tz_min / 60,

                     tz_min % 60);

        }

        avio_printf(out, "#EXT-X-PROGRAM-DATE-TIME:%s.%03d%s\n", buf0, milli, buf1);

        *prog_date_time += duration;

    }

    if (baseurl)

        avio_printf(out, "%s", baseurl);

    avio_printf(out, "%s\n", filename);



    return 0;

}
