static void writer_print_time(WriterContext *wctx, const char *key,

                              int64_t ts, const AVRational *time_base, int is_duration)

{

    char buf[128];



    if ((!is_duration && ts == AV_NOPTS_VALUE) || (is_duration && ts == 0)) {

        writer_print_string(wctx, key, "N/A", 1);

    } else {

        double d = ts * av_q2d(*time_base);

        struct unit_value uv;

        uv.val.d = d;

        uv.unit = unit_second_str;

        value_string(buf, sizeof(buf), uv);

        writer_print_string(wctx, key, buf, 0);

    }

}
