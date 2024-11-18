static int qemu_chr_open_pty(QemuOpts *opts, CharDriverState **_chr)

{

    CharDriverState *chr;

    PtyCharDriver *s;

    struct termios tty;

    int slave_fd, len;

#if defined(__OpenBSD__) || defined(__DragonFly__)

    char pty_name[PATH_MAX];

#define q_ptsname(x) pty_name

#else

    char *pty_name = NULL;

#define q_ptsname(x) ptsname(x)

#endif



    chr = g_malloc0(sizeof(CharDriverState));

    s = g_malloc0(sizeof(PtyCharDriver));



    if (openpty(&s->fd, &slave_fd, pty_name, NULL, NULL) < 0) {

        return -errno;

    }



    /* Set raw attributes on the pty. */

    tcgetattr(slave_fd, &tty);

    cfmakeraw(&tty);

    tcsetattr(slave_fd, TCSAFLUSH, &tty);

    close(slave_fd);



    len = strlen(q_ptsname(s->fd)) + 5;

    chr->filename = g_malloc(len);

    snprintf(chr->filename, len, "pty:%s", q_ptsname(s->fd));

    qemu_opt_set(opts, "path", q_ptsname(s->fd));

    fprintf(stderr, "char device redirected to %s\n", q_ptsname(s->fd));



    chr->opaque = s;

    chr->chr_write = pty_chr_write;

    chr->chr_update_read_handler = pty_chr_update_read_handler;

    chr->chr_close = pty_chr_close;



    s->timer = qemu_new_timer_ms(rt_clock, pty_chr_timer, chr);



    *_chr = chr;

    return 0;

}
