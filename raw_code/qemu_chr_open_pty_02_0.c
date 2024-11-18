static CharDriverState *qemu_chr_open_pty(void)

{

    struct termios tty;

    int master_fd, slave_fd;



    if (openpty(&master_fd, &slave_fd, NULL, NULL, NULL) < 0) {

        return NULL;

    }



    /* Set raw attributes on the pty. */

    cfmakeraw(&tty);

    tcsetattr(slave_fd, TCSAFLUSH, &tty);



    fprintf(stderr, "char device redirected to %s\n", ptsname(master_fd));

    return qemu_chr_open_fd(master_fd, master_fd);

}
