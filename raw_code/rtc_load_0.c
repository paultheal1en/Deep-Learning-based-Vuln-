static int rtc_load(QEMUFile *f, void *opaque, int version_id)

{

    RTCState *s = opaque;



    if (version_id != 1)

        return -EINVAL;



    qemu_get_buffer(f, s->cmos_data, 128);

    qemu_get_8s(f, &s->cmos_index);



    s->current_tm.tm_sec=qemu_get_be32(f);

    s->current_tm.tm_min=qemu_get_be32(f);

    s->current_tm.tm_hour=qemu_get_be32(f);

    s->current_tm.tm_wday=qemu_get_be32(f);

    s->current_tm.tm_mday=qemu_get_be32(f);

    s->current_tm.tm_mon=qemu_get_be32(f);

    s->current_tm.tm_year=qemu_get_be32(f);



    qemu_get_timer(f, s->periodic_timer);

    s->next_periodic_time=qemu_get_be64(f);



    s->next_second_time=qemu_get_be64(f);

    qemu_get_timer(f, s->second_timer);

    qemu_get_timer(f, s->second_timer2);

    return 0;

}
