static void fd_chr_close(struct CharDriverState *chr)

{

    FDCharDriver *s = chr->opaque;



    if (s->fd_in_tag) {

        g_source_remove(s->fd_in_tag);

        s->fd_in_tag = 0;

    }



    if (s->fd_in) {

        g_io_channel_unref(s->fd_in);

    }

    if (s->fd_out) {

        g_io_channel_unref(s->fd_out);

    }



    g_free(s);

    qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

}
