static void char_socket_finalize(Object *obj)

{

    Chardev *chr = CHARDEV(obj);

    SocketChardev *s = SOCKET_CHARDEV(obj);



    tcp_chr_free_connection(chr);



    if (s->reconnect_timer) {

        g_source_remove(s->reconnect_timer);

        s->reconnect_timer = 0;

    }

    qapi_free_SocketAddressLegacy(s->addr);

    if (s->listen_tag) {

        g_source_remove(s->listen_tag);

        s->listen_tag = 0;

    }

    if (s->listen_ioc) {

        object_unref(OBJECT(s->listen_ioc));

    }

    if (s->tls_creds) {

        object_unref(OBJECT(s->tls_creds));

    }



    qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

}
