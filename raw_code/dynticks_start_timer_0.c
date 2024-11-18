static int dynticks_start_timer(struct qemu_alarm_timer *t)

{

    struct sigevent ev;

    timer_t host_timer;

    struct sigaction act;



    sigfillset(&act.sa_mask);

    act.sa_flags = 0;

    act.sa_handler = host_alarm_handler;



    sigaction(SIGALRM, &act, NULL);



    /* 

     * Initialize ev struct to 0 to avoid valgrind complaining

     * about uninitialized data in timer_create call

     */

    memset(&ev, 0, sizeof(ev));

    ev.sigev_value.sival_int = 0;

    ev.sigev_notify = SIGEV_SIGNAL;

#ifdef SIGEV_THREAD_ID

    if (qemu_signalfd_available()) {

        ev.sigev_notify = SIGEV_THREAD_ID;

        ev._sigev_un._tid = qemu_get_thread_id();

    }

#endif /* SIGEV_THREAD_ID */

    ev.sigev_signo = SIGALRM;



    if (timer_create(CLOCK_REALTIME, &ev, &host_timer)) {

        perror("timer_create");

        return -1;

    }



    t->timer = host_timer;



    return 0;

}
