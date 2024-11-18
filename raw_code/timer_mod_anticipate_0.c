void timer_mod_anticipate(QEMUTimer *ts, int64_t expire_time)

{

    timer_mod_anticipate_ns(ts, expire_time * ts->scale);

}
