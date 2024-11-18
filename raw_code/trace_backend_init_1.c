bool trace_backend_init(const char *events, const char *file)

{

    GThread *thread;



    if (!g_thread_supported()) {


        g_thread_init(NULL);





    }



    trace_available_cond = g_cond_new();

    trace_empty_cond = g_cond_new();



    thread = trace_thread_create(writeout_thread);

    if (!thread) {

        fprintf(stderr, "warning: unable to initialize simple trace backend\n");

        return false;

    }



    atexit(st_flush_trace_buffer);

    trace_backend_init_events(events);

    st_set_trace_file(file);

    return true;

}