static void char_socket_test(void)

{

    Chardev *chr = qemu_chr_new("server", "tcp:127.0.0.1:0,server,nowait");

    Chardev *chr_client;

    QObject *addr;

    QDict *qdict, *data;

    const char *port;

    SocketIdleData d = { .chr = chr };

    CharBackend be;

    CharBackend client_be;

    char *tmp;



    d.be = &be;

    d.client_be = &be;



    g_assert_nonnull(chr);

    g_assert(!object_property_get_bool(OBJECT(chr), "connected", &error_abort));



    addr = object_property_get_qobject(OBJECT(chr), "addr", &error_abort);

    qdict = qobject_to_qdict(addr);

    data = qdict_get_qdict(qdict, "data");

    port = qdict_get_str(data, "port");

    tmp = g_strdup_printf("tcp:127.0.0.1:%s", port);

    QDECREF(qdict);



    qemu_chr_fe_init(&be, chr, &error_abort);

    qemu_chr_fe_set_handlers(&be, socket_can_read, socket_read,

                             NULL, &d, NULL, true);



    chr_client = qemu_chr_new("client", tmp);

    qemu_chr_fe_init(&client_be, chr_client, &error_abort);

    qemu_chr_fe_set_handlers(&client_be, socket_can_read_hello,

                             socket_read_hello,

                             NULL, &d, NULL, true);

    g_free(tmp);



    d.conn_expected = true;

    guint id = g_idle_add(char_socket_test_idle, &d);

    g_source_set_name_by_id(id, "test-idle");

    g_assert_cmpint(id, >, 0);

    main_loop();



    g_assert(object_property_get_bool(OBJECT(chr), "connected", &error_abort));

    g_assert(object_property_get_bool(OBJECT(chr_client),

                                      "connected", &error_abort));



    qemu_chr_write_all(chr_client, (const uint8_t *)"Z", 1);

    main_loop();



    object_unparent(OBJECT(chr_client));



    d.conn_expected = false;

    g_idle_add(char_socket_test_idle, &d);

    main_loop();



    object_unparent(OBJECT(chr));

}
