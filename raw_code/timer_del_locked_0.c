static void timer_del_locked(QEMUTimerList *timer_list, QEMUTimer *ts)

{

    QEMUTimer **pt, *t;



    ts->expire_time = -1;

    pt = &timer_list->active_timers;

    for(;;) {

        t = *pt;

        if (!t)

            break;

        if (t == ts) {

            *pt = t->next;

            break;

        }

        pt = &t->next;

    }

}
