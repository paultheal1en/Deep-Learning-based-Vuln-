static gboolean tcp_chr_chan_close(GIOChannel *channel, GIOCondition cond,

                                   void *opaque)

{

    CharDriverState *chr = opaque;



    if (cond != G_IO_HUP) {

        return FALSE;

    }



    /* connection closed */

    tcp_chr_disconnect(chr);

    if (chr->fd_hup_tag) {

        g_source_remove(chr->fd_hup_tag);

        chr->fd_hup_tag = 0;

    }



    return TRUE;

}
