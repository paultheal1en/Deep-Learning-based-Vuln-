int init_timer_alarm(void)

{

    struct qemu_alarm_timer *t = NULL;

    int i, err = -1;



    for (i = 0; alarm_timers[i].name; i++) {

        t = &alarm_timers[i];



        err = t->start(t);

        if (!err)

            break;

    }



    if (err) {

        err = -ENOENT;

        goto fail;

    }



    /* first event is at time 0 */

    atexit(quit_timers);

    t->pending = true;

    alarm_timer = t;



    return 0;



fail:

    return err;

}
