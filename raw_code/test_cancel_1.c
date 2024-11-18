static void test_cancel(void)

{

    WorkerTestData data[100];

    int num_canceled;

    int i;



    /* Start more work items than there will be threads, to ensure

     * the pool is full.

     */

    test_submit_many();



    /* Start long running jobs, to ensure we can cancel some.  */

    for (i = 0; i < 100; i++) {

        data[i].n = 0;

        data[i].ret = -EINPROGRESS;

        data[i].aiocb = thread_pool_submit_aio(pool, long_cb, &data[i],

                                               done_cb, &data[i]);

    }



    /* Starting the threads may be left to a bottom half.  Let it

     * run, but do not waste too much time...

     */

    active = 100;

    aio_notify(ctx);

    aio_poll(ctx, false);



    /* Wait some time for the threads to start, with some sanity

     * testing on the behavior of the scheduler...

     */

    g_assert_cmpint(active, ==, 100);

    g_usleep(1000000);

    g_assert_cmpint(active, >, 50);



    /* Cancel the jobs that haven't been started yet.  */

    num_canceled = 0;

    for (i = 0; i < 100; i++) {

        if (atomic_cmpxchg(&data[i].n, 0, 3) == 0) {

            data[i].ret = -ECANCELED;

            bdrv_aio_cancel(data[i].aiocb);

            active--;

            num_canceled++;

        }

    }

    g_assert_cmpint(active, >, 0);

    g_assert_cmpint(num_canceled, <, 100);



    /* Canceling the others will be a blocking operation.  */

    for (i = 0; i < 100; i++) {

        if (data[i].n != 3) {

            bdrv_aio_cancel(data[i].aiocb);

        }

    }



    /* Finish execution and execute any remaining callbacks.  */

    qemu_aio_wait_all();

    g_assert_cmpint(active, ==, 0);

    for (i = 0; i < 100; i++) {

        if (data[i].n == 3) {

            g_assert_cmpint(data[i].ret, ==, -ECANCELED);

            g_assert(data[i].aiocb != NULL);

        } else {

            g_assert_cmpint(data[i].n, ==, 2);

            g_assert_cmpint(data[i].ret, ==, 0);

            g_assert(data[i].aiocb == NULL);

        }

    }

}
