bool timerlist_run_timers(QEMUTimerList *timer_list)

{

    QEMUTimer *ts;

    int64_t current_time;

    bool progress = false;

    QEMUTimerCB *cb;

    void *opaque;



    if (!atomic_read(&timer_list->active_timers)) {

        return false;

    }



    qemu_event_reset(&timer_list->timers_done_ev);

    if (!timer_list->clock->enabled) {

        goto out;

    }



    switch (timer_list->clock->type) {

    case QEMU_CLOCK_REALTIME:

        break;

    default:

    case QEMU_CLOCK_VIRTUAL:

        if (!replay_checkpoint(CHECKPOINT_CLOCK_VIRTUAL)) {

            goto out;

        }

        break;

    case QEMU_CLOCK_HOST:

        if (!replay_checkpoint(CHECKPOINT_CLOCK_HOST)) {

            goto out;

        }

        break;

    case QEMU_CLOCK_VIRTUAL_RT:

        if (!replay_checkpoint(CHECKPOINT_CLOCK_VIRTUAL_RT)) {

            goto out;

        }

        break;

    }



    current_time = qemu_clock_get_ns(timer_list->clock->type);

    for(;;) {

        qemu_mutex_lock(&timer_list->active_timers_lock);

        ts = timer_list->active_timers;

        if (!timer_expired_ns(ts, current_time)) {

            qemu_mutex_unlock(&timer_list->active_timers_lock);

            break;

        }



        /* remove timer from the list before calling the callback */

        timer_list->active_timers = ts->next;

        ts->next = NULL;

        ts->expire_time = -1;

        cb = ts->cb;

        opaque = ts->opaque;

        qemu_mutex_unlock(&timer_list->active_timers_lock);



        /* run the callback (the timer list can be modified) */

        cb(opaque);

        progress = true;

    }



out:

    qemu_event_set(&timer_list->timers_done_ev);

    return progress;

}
