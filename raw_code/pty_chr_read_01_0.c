static gboolean pty_chr_read(GIOChannel *chan, GIOCondition cond, void *opaque)

{

    CharDriverState *chr = opaque;

    PtyCharDriver *s = chr->opaque;

    gsize size, len;

    uint8_t buf[READ_BUF_LEN];

    GIOStatus status;



    len = sizeof(buf);

    if (len > s->read_bytes)

        len = s->read_bytes;

    if (len == 0)

        return FALSE;

    status = g_io_channel_read_chars(s->fd, (gchar *)buf, len, &size, NULL);

    if (status != G_IO_STATUS_NORMAL) {

        pty_chr_state(chr, 0);

        return FALSE;

    } else {

        pty_chr_state(chr, 1);

        qemu_chr_be_write(chr, buf, size);

    }

    return TRUE;

}
