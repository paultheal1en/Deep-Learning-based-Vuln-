static void test_submit_many(void)

{

    WorkerTestData data[100];

    int i;



    /* Start more work items than there will be threads.  */

    for (i = 0; i < 100; i++) {

        data[i].n = 0;

        data[i].ret = -EINPROGRESS;

        thread_pool_submit_aio(worker_cb, &data[i], done_cb, &data[i]);

    }



    active = 100;

    while (active > 0) {

        qemu_aio_wait();

    }

    for (i = 0; i < 100; i++) {

        g_assert_cmpint(data[i].n, ==, 1);

        g_assert_cmpint(data[i].ret, ==, 0);

    }

}
