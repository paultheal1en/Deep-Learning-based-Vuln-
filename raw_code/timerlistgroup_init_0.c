void timerlistgroup_init(QEMUTimerListGroup *tlg,

                         QEMUTimerListNotifyCB *cb, void *opaque)

{

    QEMUClockType type;

    for (type = 0; type < QEMU_CLOCK_MAX; type++) {

        tlg->tl[type] = timerlist_new(type, cb, opaque);

    }

}
