static void *thread_function(void *data)

{

    GMainLoop *loop;

    loop = g_main_loop_new(NULL, FALSE);

    g_main_loop_run(loop);

    return NULL;

}
