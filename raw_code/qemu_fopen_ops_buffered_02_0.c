QEMUFile *qemu_fopen_ops_buffered(void *opaque,

                                  size_t bytes_per_sec,

                                  BufferedPutFunc *put_buffer,

                                  BufferedPutReadyFunc *put_ready,

                                  BufferedWaitForUnfreezeFunc *wait_for_unfreeze,

                                  BufferedCloseFunc *close)

{

    QEMUFileBuffered *s;



    s = qemu_mallocz(sizeof(*s));



    s->opaque = opaque;

    s->xfer_limit = bytes_per_sec / 10;

    s->put_buffer = put_buffer;

    s->put_ready = put_ready;

    s->wait_for_unfreeze = wait_for_unfreeze;

    s->close = close;



    s->file = qemu_fopen_ops(s, buffered_put_buffer, NULL,

                             buffered_close, buffered_rate_limit,

                             buffered_set_rate_limit,

			     buffered_get_rate_limit);



    s->timer = qemu_new_timer(rt_clock, buffered_rate_tick, s);



    qemu_mod_timer(s->timer, qemu_get_clock(rt_clock) + 100);



    return s->file;

}
