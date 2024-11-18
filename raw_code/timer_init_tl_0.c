void timer_init_tl(QEMUTimer *ts,

                   QEMUTimerList *timer_list, int scale,

                   QEMUTimerCB *cb, void *opaque)

{

    ts->timer_list = timer_list;

    ts->cb = cb;

    ts->opaque = opaque;

    ts->scale = scale;

    ts->expire_time = -1;

}
