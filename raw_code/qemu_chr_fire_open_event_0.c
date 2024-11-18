static void qemu_chr_fire_open_event(void *opaque)

{

    CharDriverState *s = opaque;

    qemu_chr_be_event(s, CHR_EVENT_OPENED);

    qemu_free_timer(s->open_timer);

    s->open_timer = NULL;

}
