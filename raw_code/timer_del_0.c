void timer_del(QEMUTimer *ts)

{

    QEMUTimer **pt, *t;



    pt = &ts->timer_list->active_timers;

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
