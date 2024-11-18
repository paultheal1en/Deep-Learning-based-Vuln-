static void kbd_send_chars(void *opaque)

{

    TextConsole *s = opaque;

    int len;

    uint8_t buf[16];



    len = qemu_chr_can_read(s->chr);

    if (len > s->out_fifo.count)

        len = s->out_fifo.count;

    if (len > 0) {

        if (len > sizeof(buf))

            len = sizeof(buf);

        qemu_fifo_read(&s->out_fifo, buf, len);

        qemu_chr_read(s->chr, buf, len);

    }

    /* characters are pending: we send them a bit later (XXX:

       horrible, should change char device API) */

    if (s->out_fifo.count > 0) {

        qemu_mod_timer(s->kbd_timer, qemu_get_clock(rt_clock) + 1);

    }

}
