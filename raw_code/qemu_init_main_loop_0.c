int qemu_init_main_loop(Error **errp)

{

    int ret;

    GSource *src;

    Error *local_error = NULL;



    init_clocks();



    ret = qemu_signal_init();

    if (ret) {

        return ret;

    }



    qemu_aio_context = aio_context_new(&local_error);

    if (!qemu_aio_context) {

        error_propagate(errp, local_error);

        return -EMFILE;

    }

    qemu_notify_bh = qemu_bh_new(notify_event_cb, NULL);

    gpollfds = g_array_new(FALSE, FALSE, sizeof(GPollFD));

    src = aio_get_g_source(qemu_aio_context);

    g_source_set_name(src, "aio-context");

    g_source_attach(src, NULL);

    g_source_unref(src);

    src = iohandler_get_g_source();

    g_source_set_name(src, "io-handler");

    g_source_attach(src, NULL);

    g_source_unref(src);

    return 0;

}
