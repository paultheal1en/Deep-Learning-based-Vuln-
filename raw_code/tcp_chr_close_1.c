static void tcp_chr_close(CharDriverState *chr)

{

    TCPCharDriver *s = chr->opaque;

    if (s->fd >= 0) {

        if (s->tag) {

            g_source_remove(s->tag);

            s->tag = 0;

        }

        if (s->chan) {

            g_io_channel_unref(s->chan);

        }

        closesocket(s->fd);

    }

    if (s->listen_fd >= 0) {

        if (s->listen_tag) {

            g_source_remove(s->listen_tag);

            s->listen_tag = 0;

        }

        if (s->listen_chan) {

            g_io_channel_unref(s->listen_chan);

        }

        closesocket(s->listen_fd);

    }

    g_free(s);

    qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

}
