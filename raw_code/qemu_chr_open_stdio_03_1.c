static CharDriverState *qemu_chr_open_stdio(ChardevStdio *opts)

{

    CharDriverState *chr;



    if (is_daemonized()) {

        error_report("cannot use stdio with -daemonize");

        return NULL;

    }

    old_fd0_flags = fcntl(0, F_GETFL);

    tcgetattr (0, &oldtty);

    qemu_set_nonblock(0);

    atexit(term_exit);



    chr = qemu_chr_open_fd(0, 1);

    chr->chr_close = qemu_chr_close_stdio;

    chr->chr_set_echo = qemu_chr_set_echo_stdio;

    if (opts->has_signal) {

        stdio_allow_signal = opts->signal;

    }

    qemu_chr_fe_set_echo(chr, false);



    return chr;

}
