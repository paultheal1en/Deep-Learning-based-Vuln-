static int read_key(void)

{

#if defined(HAVE_CONIO_H)

    if(kbhit())

        return(getch());

#elif defined(HAVE_TERMIOS_H)

    int n = 1;

    unsigned char ch;

#ifndef CONFIG_BEOS_NETSERVER

    struct timeval tv;

    fd_set rfds;



    FD_ZERO(&rfds);

    FD_SET(0, &rfds);

    tv.tv_sec = 0;

    tv.tv_usec = 0;

    n = select(1, &rfds, NULL, NULL, &tv);

#endif

    if (n > 0) {

        n = read(0, &ch, 1);

        if (n == 1)

            return ch;



        return n;

    }

#endif

    return -1;

}
