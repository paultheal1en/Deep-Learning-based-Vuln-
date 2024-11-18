static char *ts_value_string (char *buf, int buf_size, int64_t ts)

{

    if (ts == AV_NOPTS_VALUE) {

        snprintf(buf, buf_size, "N/A");

    } else {

        snprintf(buf, buf_size, "%"PRId64, ts);

    }



    return buf;

}
