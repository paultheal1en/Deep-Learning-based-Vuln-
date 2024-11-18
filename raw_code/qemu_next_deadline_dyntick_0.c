static uint64_t qemu_next_deadline_dyntick(void)

{

    int64_t delta;

    int64_t rtdelta;



    if (use_icount)

        delta = INT32_MAX;

    else

        delta = (qemu_next_deadline() + 999) / 1000;



    if (active_timers[QEMU_TIMER_REALTIME]) {

        rtdelta = (active_timers[QEMU_TIMER_REALTIME]->expire_time -

                 qemu_get_clock(rt_clock))*1000;

        if (rtdelta < delta)

            delta = rtdelta;

    }



    if (delta < MIN_TIMER_REARM_US)

        delta = MIN_TIMER_REARM_US;



    return delta;

}
