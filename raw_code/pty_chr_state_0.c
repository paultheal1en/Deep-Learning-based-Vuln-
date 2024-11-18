static void pty_chr_state(CharDriverState *chr, int connected)

{

    PtyCharDriver *s = chr->opaque;



    if (!connected) {

        if (s->fd_tag) {

            g_source_remove(s->fd_tag);

            s->fd_tag = 0;

        }

        s->connected = 0;

        s->polling = 0;

        /* (re-)connect poll interval for idle guests: once per second.

         * We check more frequently in case the guests sends data to

         * the virtual device linked to our pty. */

        pty_chr_rearm_timer(chr, 1000);

    } else {

        if (!s->connected)

            qemu_chr_be_generic_open(chr);

        s->connected = 1;

    }

}
