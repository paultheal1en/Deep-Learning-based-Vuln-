static void test_io_channel_command_fifo(bool async)

{

#define TEST_FIFO "tests/test-io-channel-command.fifo"

    QIOChannel *src, *dst;

    QIOChannelTest *test;

    char *srcfifo = g_strdup_printf("PIPE:%s,wronly", TEST_FIFO);

    char *dstfifo = g_strdup_printf("PIPE:%s,rdonly", TEST_FIFO);

    const char *srcargv[] = {

        "/bin/socat", "-", srcfifo, NULL,

    };

    const char *dstargv[] = {

        "/bin/socat", dstfifo, "-", NULL,

    };



    unlink(TEST_FIFO);

    if (access("/bin/socat", X_OK) < 0) {

        return; /* Pretend success if socat is not present */

    }

    if (mkfifo(TEST_FIFO, 0600) < 0) {

        abort();

    }

    src = QIO_CHANNEL(qio_channel_command_new_spawn(srcargv,

                                                    O_WRONLY,

                                                    &error_abort));

    dst = QIO_CHANNEL(qio_channel_command_new_spawn(dstargv,

                                                    O_RDONLY,

                                                    &error_abort));



    test = qio_channel_test_new();

    qio_channel_test_run_threads(test, async, src, dst);

    qio_channel_test_validate(test);



    object_unref(OBJECT(src));

    object_unref(OBJECT(dst));



    g_free(srcfifo);

    g_free(dstfifo);

    unlink(TEST_FIFO);

}
