static void dynticks_rearm_timer(struct qemu_alarm_timer *t)

{

    timer_t host_timer = (timer_t)(long)t->priv;

    struct itimerspec timeout;

    int64_t nearest_delta_us = INT64_MAX;

    int64_t current_us;



    if (!active_timers[QEMU_TIMER_REALTIME] &&

                !active_timers[QEMU_TIMER_VIRTUAL])

        return;



    nearest_delta_us = qemu_next_deadline_dyntick();



    /* check whether a timer is already running */

    if (timer_gettime(host_timer, &timeout)) {

        perror("gettime");

        fprintf(stderr, "Internal timer error: aborting\n");

        exit(1);

    }

    current_us = timeout.it_value.tv_sec * 1000000 + timeout.it_value.tv_nsec/1000;

    if (current_us && current_us <= nearest_delta_us)

        return;



    timeout.it_interval.tv_sec = 0;

    timeout.it_interval.tv_nsec = 0; /* 0 for one-shot timer */

    timeout.it_value.tv_sec =  nearest_delta_us / 1000000;

    timeout.it_value.tv_nsec = (nearest_delta_us % 1000000) * 1000;

    if (timer_settime(host_timer, 0 /* RELATIVE */, &timeout, NULL)) {

        perror("settime");

        fprintf(stderr, "Internal timer error: aborting\n");

        exit(1);

    }

}
