void timer_mod_ns(QEMUTimer *ts, int64_t expire_time)

{

    QEMUTimerList *timer_list = ts->timer_list;

    bool rearm;



    qemu_mutex_lock(&timer_list->active_timers_lock);

    timer_del_locked(timer_list, ts);

    rearm = timer_mod_ns_locked(timer_list, ts, expire_time);

    qemu_mutex_unlock(&timer_list->active_timers_lock);



    if (rearm) {

        timerlist_rearm(timer_list);

    }

}
