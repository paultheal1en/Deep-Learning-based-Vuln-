static void test_io_channel_unix_fd_pass(void)

{

    SocketAddress *listen_addr = g_new0(SocketAddress, 1);

    SocketAddress *connect_addr = g_new0(SocketAddress, 1);

    QIOChannel *src, *dst;

    int testfd;

    int fdsend[3];

    int *fdrecv = NULL;

    size_t nfdrecv = 0;

    size_t i;

    char bufsend[12], bufrecv[12];

    struct iovec iosend[1], iorecv[1];



#define TEST_SOCKET "test-io-channel-socket.sock"

#define TEST_FILE "test-io-channel-socket.txt"



    testfd = open(TEST_FILE, O_RDWR|O_TRUNC|O_CREAT, 0700);

    g_assert(testfd != -1);

    fdsend[0] = testfd;

    fdsend[1] = testfd;

    fdsend[2] = testfd;



    listen_addr->type = SOCKET_ADDRESS_KIND_UNIX;

    listen_addr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

    listen_addr->u.q_unix.data->path = g_strdup(TEST_SOCKET);



    connect_addr->type = SOCKET_ADDRESS_KIND_UNIX;

    connect_addr->u.q_unix.data = g_new0(UnixSocketAddress, 1);

    connect_addr->u.q_unix.data->path = g_strdup(TEST_SOCKET);



    test_io_channel_setup_sync(listen_addr, connect_addr, &src, &dst);



    memcpy(bufsend, "Hello World", G_N_ELEMENTS(bufsend));



    iosend[0].iov_base = bufsend;

    iosend[0].iov_len = G_N_ELEMENTS(bufsend);



    iorecv[0].iov_base = bufrecv;

    iorecv[0].iov_len = G_N_ELEMENTS(bufrecv);



    g_assert(qio_channel_has_feature(src, QIO_CHANNEL_FEATURE_FD_PASS));

    g_assert(qio_channel_has_feature(dst, QIO_CHANNEL_FEATURE_FD_PASS));



    qio_channel_writev_full(src,

                            iosend,

                            G_N_ELEMENTS(iosend),

                            fdsend,

                            G_N_ELEMENTS(fdsend),

                            &error_abort);



    qio_channel_readv_full(dst,

                           iorecv,

                           G_N_ELEMENTS(iorecv),

                           &fdrecv,

                           &nfdrecv,

                           &error_abort);



    g_assert(nfdrecv == G_N_ELEMENTS(fdsend));

    /* Each recvd FD should be different from sent FD */

    for (i = 0; i < nfdrecv; i++) {

        g_assert_cmpint(fdrecv[i], !=, testfd);

    }

    /* Each recvd FD should be different from each other */

    g_assert_cmpint(fdrecv[0], !=, fdrecv[1]);

    g_assert_cmpint(fdrecv[0], !=, fdrecv[2]);

    g_assert_cmpint(fdrecv[1], !=, fdrecv[2]);



    /* Check the I/O buf we sent at the same time matches */

    g_assert(memcmp(bufsend, bufrecv, G_N_ELEMENTS(bufsend)) == 0);



    /* Write some data into the FD we received */

    g_assert(write(fdrecv[0], bufsend, G_N_ELEMENTS(bufsend)) ==

             G_N_ELEMENTS(bufsend));



    /* Read data from the original FD and make sure it matches */

    memset(bufrecv, 0, G_N_ELEMENTS(bufrecv));

    g_assert(lseek(testfd, 0, SEEK_SET) == 0);

    g_assert(read(testfd, bufrecv, G_N_ELEMENTS(bufrecv)) ==

             G_N_ELEMENTS(bufrecv));

    g_assert(memcmp(bufsend, bufrecv, G_N_ELEMENTS(bufsend)) == 0);



    object_unref(OBJECT(src));

    object_unref(OBJECT(dst));

    qapi_free_SocketAddress(listen_addr);

    qapi_free_SocketAddress(connect_addr);

    unlink(TEST_SOCKET);

    unlink(TEST_FILE);

    close(testfd);

    for (i = 0; i < nfdrecv; i++) {

        close(fdrecv[i]);

    }

    g_free(fdrecv);

}
