static void char_socket_class_init(ObjectClass *oc, void *data)

{

    ChardevClass *cc = CHARDEV_CLASS(oc);



    cc->parse = qemu_chr_parse_socket;

    cc->open = qmp_chardev_open_socket;

    cc->chr_wait_connected = tcp_chr_wait_connected;

    cc->chr_write = tcp_chr_write;

    cc->chr_sync_read = tcp_chr_sync_read;

    cc->chr_disconnect = tcp_chr_disconnect;

    cc->get_msgfds = tcp_get_msgfds;

    cc->set_msgfds = tcp_set_msgfds;

    cc->chr_add_client = tcp_chr_add_client;

    cc->chr_add_watch = tcp_chr_add_watch;

    cc->chr_update_read_handler = tcp_chr_update_read_handler;



    object_class_property_add(oc, "addr", "SocketAddressLegacy",

                              char_socket_get_addr, NULL,

                              NULL, NULL, &error_abort);



    object_class_property_add_bool(oc, "connected", char_socket_get_connected,

                                   NULL, &error_abort);

}
