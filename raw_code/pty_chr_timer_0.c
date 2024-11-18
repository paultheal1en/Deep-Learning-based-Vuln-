static gboolean pty_chr_timer(gpointer opaque)

{

    struct CharDriverState *chr = opaque;

    PtyCharDriver *s = chr->opaque;



    if (s->connected) {

        goto out;

    }

    if (s->polling) {

        /* If we arrive here without polling being cleared due

         * read returning -EIO, then we are (re-)connected */

        pty_chr_state(chr, 1);

        goto out;

    }



    /* Next poll ... */

    pty_chr_update_read_handler(chr);



out:

    return FALSE;

}
