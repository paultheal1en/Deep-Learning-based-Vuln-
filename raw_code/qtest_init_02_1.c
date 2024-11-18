QTestState *qtest_init(const char *extra_args)

{

    QTestState *s;

    int sock, qmpsock, i;

    gchar *socket_path;

    gchar *qmp_socket_path;

    gchar *command;

    const char *qemu_binary;

    struct sigaction sigact;



    qemu_binary = getenv("QTEST_QEMU_BINARY");

    g_assert(qemu_binary != NULL);



    s = g_malloc(sizeof(*s));



    socket_path = g_strdup_printf("/tmp/qtest-%d.sock", getpid());

    qmp_socket_path = g_strdup_printf("/tmp/qtest-%d.qmp", getpid());



    sock = init_socket(socket_path);

    qmpsock = init_socket(qmp_socket_path);



    /* Catch SIGABRT to clean up on g_assert() failure */

    sigact = (struct sigaction){

        .sa_handler = sigabrt_handler,

        .sa_flags = SA_RESETHAND,

    };

    sigemptyset(&sigact.sa_mask);

    sigaction(SIGABRT, &sigact, &s->sigact_old);



    s->qemu_pid = fork();

    if (s->qemu_pid == 0) {

        command = g_strdup_printf("exec %s "

                                  "-qtest unix:%s,nowait "

                                  "-qtest-log /dev/null "

                                  "-qmp unix:%s,nowait "

                                  "-machine accel=qtest "

                                  "-display none "

                                  "%s", qemu_binary, socket_path,

                                  qmp_socket_path,

                                  extra_args ?: "");

        execlp("/bin/sh", "sh", "-c", command, NULL);

        exit(1);

    }



    s->fd = socket_accept(sock);

    s->qmp_fd = socket_accept(qmpsock);

    unlink(socket_path);

    unlink(qmp_socket_path);

    g_free(socket_path);

    g_free(qmp_socket_path);



    s->rx = g_string_new("");

    for (i = 0; i < MAX_IRQ; i++) {

        s->irq_level[i] = false;

    }



    /* Read the QMP greeting and then do the handshake */

    qtest_qmp_discard_response(s, "");

    qtest_qmp_discard_response(s, "{ 'execute': 'qmp_capabilities' }");



    if (getenv("QTEST_STOP")) {

        kill(s->qemu_pid, SIGSTOP);

    }



    return s;

}
