static void test_source_notify(void)

{

    while (g_main_context_iteration(NULL, false));

    aio_notify(ctx);

    g_assert(g_main_context_iteration(NULL, true));

    g_assert(!g_main_context_iteration(NULL, false));

}
