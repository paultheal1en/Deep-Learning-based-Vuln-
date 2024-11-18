static char *time_value_string(char *buf, int buf_size, int64_t val, const AVRational *time_base)

{

    if (val == AV_NOPTS_VALUE) {

        snprintf(buf, buf_size, "N/A");

    } else {

        double d = val * av_q2d(*time_base);

        value_string(buf, buf_size, (struct unit_value){.val.d=d, .unit=unit_second_str});

    }



    return buf;

}
