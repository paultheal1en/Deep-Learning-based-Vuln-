static void term_init(void)

{

    struct termios tty;



    tcgetattr (0, &tty);

    oldtty = tty;



    tty.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP

                          |INLCR|IGNCR|ICRNL|IXON);

    tty.c_oflag |= OPOST;

    tty.c_lflag &= ~(ECHO|ECHONL|ICANON|IEXTEN);

    tty.c_cflag &= ~(CSIZE|PARENB);

    tty.c_cflag |= CS8;

    tty.c_cc[VMIN] = 1;

    tty.c_cc[VTIME] = 0;



    tcsetattr (0, TCSANOW, &tty);



    atexit(term_exit);

}
