static void test_bh_delete_from_cb_many(void)

{

    BHTestData data1 = { .n = 0, .max = 1 };

    BHTestData data2 = { .n = 0, .max = 3 };

    BHTestData data3 = { .n = 0, .max = 2 };

    BHTestData data4 = { .n = 0, .max = 4 };



    data1.bh = aio_bh_new(ctx, bh_delete_cb, &data1);

    data2.bh = aio_bh_new(ctx, bh_delete_cb, &data2);

    data3.bh = aio_bh_new(ctx, bh_delete_cb, &data3);

    data4.bh = aio_bh_new(ctx, bh_delete_cb, &data4);



    qemu_bh_schedule(data1.bh);

    qemu_bh_schedule(data2.bh);

    qemu_bh_schedule(data3.bh);

    qemu_bh_schedule(data4.bh);

    g_assert_cmpint(data1.n, ==, 0);

    g_assert_cmpint(data2.n, ==, 0);

    g_assert_cmpint(data3.n, ==, 0);

    g_assert_cmpint(data4.n, ==, 0);



    g_assert(aio_poll(ctx, false));

    g_assert_cmpint(data1.n, ==, 1);

    g_assert_cmpint(data2.n, ==, 1);

    g_assert_cmpint(data3.n, ==, 1);

    g_assert_cmpint(data4.n, ==, 1);

    g_assert(data1.bh == NULL);



    wait_for_aio();

    g_assert_cmpint(data1.n, ==, data1.max);

    g_assert_cmpint(data2.n, ==, data2.max);

    g_assert_cmpint(data3.n, ==, data3.max);

    g_assert_cmpint(data4.n, ==, data4.max);

    g_assert(data1.bh == NULL);

    g_assert(data2.bh == NULL);

    g_assert(data3.bh == NULL);

    g_assert(data4.bh == NULL);

}
