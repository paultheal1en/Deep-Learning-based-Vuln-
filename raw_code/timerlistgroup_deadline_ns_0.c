int64_t timerlistgroup_deadline_ns(QEMUTimerListGroup *tlg)

{

    int64_t deadline = -1;

    QEMUClockType type;

    bool play = replay_mode == REPLAY_MODE_PLAY;

    for (type = 0; type < QEMU_CLOCK_MAX; type++) {

        if (qemu_clock_use_for_deadline(type)) {

            if (!play || type == QEMU_CLOCK_REALTIME) {

                deadline = qemu_soonest_timeout(deadline,

                                                timerlist_deadline_ns(tlg->tl[type]));

            } else {

                /* Read clock from the replay file and

                   do not calculate the deadline, based on virtual clock. */

                qemu_clock_get_ns(type);

            }

        }

    }

    return deadline;

}
