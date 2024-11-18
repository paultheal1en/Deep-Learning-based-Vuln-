static void pty_chr_close(struct CharDriverState *chr)

{

    PtyCharDriver *s = chr->opaque;

    int fd;



    remove_fd_in_watch(chr);

    fd = g_io_channel_unix_get_fd(s->fd);

    g_io_channel_unref(s->fd);

    close(fd);

    if (s->timer_tag) {

        g_source_remove(s->timer_tag);

        s->timer_tag = 0;

    }

    g_free(s);

    qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

}
