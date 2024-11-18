static CharDriverState *qemu_chr_open_ringbuf(const char *id,

                                              ChardevBackend *backend,

                                              ChardevReturn *ret,

                                              Error **errp)

{

    ChardevRingbuf *opts = backend->u.ringbuf;

    CharDriverState *chr;

    RingBufCharDriver *d;



    chr = qemu_chr_alloc();

    d = g_malloc(sizeof(*d));



    d->size = opts->has_size ? opts->size : 65536;



    /* The size must be power of 2 */

    if (d->size & (d->size - 1)) {

        error_setg(errp, "size of ringbuf chardev must be power of two");

        goto fail;

    }



    d->prod = 0;

    d->cons = 0;

    d->cbuf = g_malloc0(d->size);



    chr->opaque = d;

    chr->chr_write = ringbuf_chr_write;

    chr->chr_close = ringbuf_chr_close;



    return chr;



fail:

    g_free(d);

    g_free(chr);

    return NULL;

}
