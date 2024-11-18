static void qemu_init_child_watch(void)

{

    struct sigaction act;

    sigchld_bh = qemu_bh_new(sigchld_bh_handler, NULL);



    memset(&act, 0, sizeof(act));

    act.sa_handler = sigchld_handler;

    act.sa_flags = SA_NOCLDSTOP;

    sigaction(SIGCHLD, &act, NULL);

}
