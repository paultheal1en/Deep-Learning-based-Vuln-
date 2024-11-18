static SpiceTimer *timer_add(SpiceTimerFunc func, void *opaque)

{

    SpiceTimer *timer;



    timer = qemu_mallocz(sizeof(*timer));

    timer->timer = qemu_new_timer(rt_clock, func, opaque);

    QTAILQ_INSERT_TAIL(&timers, timer, next);

    return timer;

}
