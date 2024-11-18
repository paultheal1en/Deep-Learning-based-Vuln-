static void buffered_rate_tick(void *opaque)

{

    QEMUFileBuffered *s = opaque;



    if (s->has_error) {

        buffered_close(s);

        return;

    }



    qemu_mod_timer(s->timer, qemu_get_clock(rt_clock) + 100);



    if (s->freeze_output)

        return;



    s->bytes_xfer = 0;



    buffered_flush(s);



    /* Add some checks around this */

    s->put_ready(s->opaque);

}
