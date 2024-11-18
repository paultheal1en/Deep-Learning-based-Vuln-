int gdbserver_start(int port)

{

    gdbserver_fd = gdbserver_open(port);

    if (gdbserver_fd < 0)

        return -1;

    /* accept connections */

    gdb_accept (NULL);

    return 0;

}
