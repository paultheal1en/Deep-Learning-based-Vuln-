void qemu_get_timer(QEMUFile *f, QEMUTimer *ts)

{

    uint64_t expire_time;



    expire_time = qemu_get_be64(f);

    if (expire_time != -1) {

        qemu_mod_timer(ts, expire_time);

    } else {

        qemu_del_timer(ts);

    }

}
