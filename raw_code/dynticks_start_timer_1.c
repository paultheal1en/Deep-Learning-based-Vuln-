static int dynticks_start_timer(struct qemu_alarm_timer *t)

{

    struct sigevent ev;

    timer_t host_timer;

    struct sigaction act;



    sigfillset(&act.sa_mask);

    act.sa_flags = 0;

    act.sa_handler = host_alarm_handler;



    sigaction(SIGALRM, &act, NULL);








    ev.sigev_value.sival_int = 0;

    ev.sigev_notify = SIGEV_SIGNAL;

    ev.sigev_signo = SIGALRM;



    if (timer_create(CLOCK_REALTIME, &ev, &host_timer)) {

        perror("timer_create");



        /* disable dynticks */

        fprintf(stderr, "Dynamic Ticks disabled\n");



        return -1;

    }



    t->priv = (void *)(long)host_timer;



    return 0;

}