static gboolean fd_chr_read(GIOChannel *chan, GIOCondition cond, void *opaque)

{

    CharDriverState *chr = opaque;

    FDCharDriver *s = chr->opaque;

    int len;

    uint8_t buf[READ_BUF_LEN];

    GIOStatus status;

    gsize bytes_read;



    len = sizeof(buf);

    if (len > s->max_size) {

        len = s->max_size;

    }

    if (len == 0) {

        return TRUE;

    }



    status = g_io_channel_read_chars(chan, (gchar *)buf,

                                     len, &bytes_read, NULL);

    if (status == G_IO_STATUS_EOF) {

        if (s->fd_in_tag) {

            g_source_remove(s->fd_in_tag);

            s->fd_in_tag = 0;

        }

        qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

        return FALSE;

    }

    if (status == G_IO_STATUS_NORMAL) {

        qemu_chr_be_write(chr, buf, bytes_read);

    }



    return TRUE;

}
