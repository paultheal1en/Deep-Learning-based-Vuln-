static void term_exit(void)

{

#ifndef __MINGW32__

    tcsetattr (0, TCSANOW, &oldtty);

#endif

}
