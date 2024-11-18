int64_t qemu_clock_deadline_ns_all(QEMUClockType type)

{

    int64_t deadline = -1;

    QEMUTimerList *timer_list;

    QEMUClock *clock = qemu_clock_ptr(type);

    QLIST_FOREACH(timer_list, &clock->timerlists, list) {

        deadline = qemu_soonest_timeout(deadline,

                                        timerlist_deadline_ns(timer_list));

    }

    return deadline;

}
