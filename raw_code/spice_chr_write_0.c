static int spice_chr_write(CharDriverState *chr, const uint8_t *buf, int len)

{

    SpiceCharDriver *s = chr->opaque;



    vmc_register_interface(s);

    assert(s->datalen == 0);

    if (s->bufsize < len) {

        s->bufsize = len;

        s->buffer = g_realloc(s->buffer, s->bufsize);

    }

    memcpy(s->buffer, buf, len);

    s->datapos = s->buffer;

    s->datalen = len;

    spice_server_char_device_wakeup(&s->sin);

    return len;

}
