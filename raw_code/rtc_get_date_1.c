static void rtc_get_date(Object *obj, Visitor *v, void *opaque,

                         const char *name, Error **errp)

{

    Error *err = NULL;

    RTCState *s = MC146818_RTC(obj);

    struct tm current_tm;



    rtc_update_time(s);

    rtc_get_time(s, &current_tm);

    visit_start_struct(v, NULL, "struct tm", name, 0, &err);

    if (err) {

        goto out;

    }

    visit_type_int32(v, &current_tm.tm_year, "tm_year", &err);

    visit_type_int32(v, &current_tm.tm_mon, "tm_mon", &err);

    visit_type_int32(v, &current_tm.tm_mday, "tm_mday", &err);

    visit_type_int32(v, &current_tm.tm_hour, "tm_hour", &err);

    visit_type_int32(v, &current_tm.tm_min, "tm_min", &err);

    visit_type_int32(v, &current_tm.tm_sec, "tm_sec", &err);

    visit_end_struct(v, &err);



out:

    error_propagate(errp, err);

}
