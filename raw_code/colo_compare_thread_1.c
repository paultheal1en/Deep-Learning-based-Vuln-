static void *colo_compare_thread(void *opaque)

{

    GMainContext *worker_context;

    GMainLoop *compare_loop;

    CompareState *s = opaque;

    GSource *timeout_source;



    worker_context = g_main_context_new();



    qemu_chr_fe_set_handlers(&s->chr_pri_in, compare_chr_can_read,

                             compare_pri_chr_in, NULL, s, worker_context, true);

    qemu_chr_fe_set_handlers(&s->chr_sec_in, compare_chr_can_read,

                             compare_sec_chr_in, NULL, s, worker_context, true);



    compare_loop = g_main_loop_new(worker_context, FALSE);



    /* To kick any packets that the secondary doesn't match */

    timeout_source = g_timeout_source_new(REGULAR_PACKET_CHECK_MS);

    g_source_set_callback(timeout_source,

                          (GSourceFunc)check_old_packet_regular, s, NULL);

    g_source_attach(timeout_source, worker_context);



    g_main_loop_run(compare_loop);



    g_source_unref(timeout_source);

    g_main_loop_unref(compare_loop);

    g_main_context_unref(worker_context);

    return NULL;

}
