static void test_ivshmem_server(void)

{

    IVState state1, state2, *s1, *s2;

    ServerThread thread;

    IvshmemServer server;

    int ret, vm1, vm2;

    int nvectors = 2;

    guint64 end_time = g_get_monotonic_time() + 5 * G_TIME_SPAN_SECOND;



    memset(tmpshmem, 0x42, TMPSHMSIZE);

    ret = ivshmem_server_init(&server, tmpserver, tmpshm,

                              TMPSHMSIZE, nvectors,

                              g_test_verbose());

    g_assert_cmpint(ret, ==, 0);



    ret = ivshmem_server_start(&server);

    g_assert_cmpint(ret, ==, 0);



    setup_vm_with_server(&state1, nvectors);

    s1 = &state1;

    setup_vm_with_server(&state2, nvectors);

    s2 = &state2;



    g_assert_cmpuint(in_reg(s1, IVPOSITION), ==, 0xffffffff);

    g_assert_cmpuint(in_reg(s2, IVPOSITION), ==, 0xffffffff);



    g_assert_cmpuint(qtest_readb(s1->qtest, (uintptr_t)s1->mem_base), ==, 0x00);



    thread.server = &server;

    ret = pipe(thread.pipe);

    g_assert_cmpint(ret, ==, 0);

    thread.thread = g_thread_new("ivshmem-server", server_thread, &thread);

    g_assert(thread.thread != NULL);



    /* waiting until mapping is done */

    while (g_get_monotonic_time() < end_time) {

        g_usleep(1000);



        if (qtest_readb(s1->qtest, (uintptr_t)s1->mem_base) == 0x42 &&

            qtest_readb(s2->qtest, (uintptr_t)s2->mem_base) == 0x42) {

            break;

        }

    }



    /* check got different VM ids */

    vm1 = in_reg(s1, IVPOSITION);

    vm2 = in_reg(s2, IVPOSITION);

    g_assert_cmpuint(vm1, !=, vm2);



    global_qtest = s1->qtest;

    ret = qpci_msix_table_size(s1->dev);

    g_assert_cmpuint(ret, ==, nvectors);



    /* ping vm2 -> vm1 */

    ret = qpci_msix_pending(s1->dev, 0);

    g_assert_cmpuint(ret, ==, 0);

    out_reg(s2, DOORBELL, vm1 << 16);

    do {

        g_usleep(10000);

        ret = qpci_msix_pending(s1->dev, 0);

    } while (ret == 0 && g_get_monotonic_time() < end_time);

    g_assert_cmpuint(ret, !=, 0);



    /* ping vm1 -> vm2 */

    global_qtest = s2->qtest;

    ret = qpci_msix_pending(s2->dev, 0);

    g_assert_cmpuint(ret, ==, 0);

    out_reg(s1, DOORBELL, vm2 << 16);

    do {

        g_usleep(10000);

        ret = qpci_msix_pending(s2->dev, 0);

    } while (ret == 0 && g_get_monotonic_time() < end_time);

    g_assert_cmpuint(ret, !=, 0);



    qtest_quit(s2->qtest);

    qtest_quit(s1->qtest);



    if (qemu_write_full(thread.pipe[1], "q", 1) != 1) {

        g_error("qemu_write_full: %s", g_strerror(errno));

    }



    g_thread_join(thread.thread);



    ivshmem_server_close(&server);

    close(thread.pipe[1]);

    close(thread.pipe[0]);

}
