static void udp_chr_update_read_handler(CharDriverState *chr)

{

    NetCharDriver *s = chr->opaque;



    if (s->tag) {

        g_source_remove(s->tag);

        s->tag = 0;

    }



    if (s->chan) {

        s->tag = io_add_watch_poll(s->chan, udp_chr_read_poll, udp_chr_read, chr);

    }

}
