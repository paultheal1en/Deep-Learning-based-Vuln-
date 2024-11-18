void qemu_clock_notify(QEMUClockType type)

{

    QEMUTimerList *timer_list;

    QEMUClock *clock = qemu_clock_ptr(type);

    QLIST_FOREACH(timer_list, &clock->timerlists, list) {

        timerlist_notify(timer_list);

    }

}
