static void test_source_wait_event_notifier(void)

{

    EventNotifierTestData data = { .n = 0, .active = 1 };

    event_notifier_init(&data.e, false);

    aio_set_event_notifier(ctx, &data.e, event_ready_cb);

    g_assert(g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 0);

    g_assert_cmpint(data.active, ==, 1);



    event_notifier_set(&data.e);

    g_assert(g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 1);

    g_assert_cmpint(data.active, ==, 0);



    while (g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 1);

    g_assert_cmpint(data.active, ==, 0);



    aio_set_event_notifier(ctx, &data.e, NULL);

    while (g_main_context_iteration(NULL, false));

    g_assert_cmpint(data.n, ==, 1);



    event_notifier_cleanup(&data.e);

}
