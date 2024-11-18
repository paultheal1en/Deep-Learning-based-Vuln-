static void unblock_io_signals(void)

{

    sigset_t set;



    sigemptyset(&set);

    sigaddset(&set, SIGUSR2);

    sigaddset(&set, SIGIO);

    sigaddset(&set, SIGALRM);

    pthread_sigmask(SIG_UNBLOCK, &set, NULL);



    sigemptyset(&set);

    sigaddset(&set, SIGUSR1);

    pthread_sigmask(SIG_BLOCK, &set, NULL);

}
