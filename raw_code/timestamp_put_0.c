static void timestamp_put(QDict *qdict)

{

    int err;

    QObject *obj;

    qemu_timeval tv;



    err = qemu_gettimeofday(&tv);

    if (err < 0)

        return;



    obj = qobject_from_jsonf("{ 'seconds': %" PRId64 ", "

                                "'microseconds': %" PRId64 " }",

                                (int64_t) tv.tv_sec, (int64_t) tv.tv_usec);

    assert(obj != NULL);



    qdict_put_obj(qdict, "timestamp", obj);

}
