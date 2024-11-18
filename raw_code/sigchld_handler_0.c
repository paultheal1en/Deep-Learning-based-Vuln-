static void sigchld_handler(int signal)

{

    qemu_bh_schedule(sigchld_bh);

}
