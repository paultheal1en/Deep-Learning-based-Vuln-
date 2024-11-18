static void test_submit_co(void)

{

    WorkerTestData data;

    Coroutine *co = qemu_coroutine_create(co_test_cb);



    qemu_coroutine_enter(co, &data);



    /* Back here once the worker has started.  */



    g_assert_cmpint(active, ==, 1);

    g_assert_cmpint(data.ret, ==, -EINPROGRESS);



    /* qemu_aio_wait_all will execute the rest of the coroutine.  */



    qemu_aio_wait_all();



    /* Back here after the coroutine has finished.  */



    g_assert_cmpint(active, ==, 0);

    g_assert_cmpint(data.ret, ==, 0);

}
