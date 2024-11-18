static void rtc_save_td(QEMUFile *f, void *opaque)

{

    RTCState *s = opaque;



    qemu_put_be32(f, s->irq_coalesced);

    qemu_put_be32(f, s->period);

}
