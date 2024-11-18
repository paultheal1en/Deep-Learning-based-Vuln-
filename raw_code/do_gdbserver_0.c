static void do_gdbserver(int argc, const char **argv)

{

    int port;



    port = DEFAULT_GDBSTUB_PORT;

    if (argc >= 2)

        port = atoi(argv[1]);

    if (gdbserver_start(port) < 0) {

        qemu_printf("Could not open gdbserver socket on port %d\n", port);

    } else {

        qemu_printf("Waiting gdb connection on port %d\n", port);

    }

}
