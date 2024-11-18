int64_t qemu_clock_get_ns(QEMUClockType type)

{

    int64_t now, last;

    QEMUClock *clock = qemu_clock_ptr(type);



    switch (type) {

    case QEMU_CLOCK_REALTIME:

        return get_clock();

    default:

    case QEMU_CLOCK_VIRTUAL:

        if (use_icount) {

            return cpu_get_icount();

        } else {

            return cpu_get_clock();

        }

    case QEMU_CLOCK_HOST:

        now = get_clock_realtime();

        last = clock->last;

        clock->last = now;

        if (now < last) {

            notifier_list_notify(&clock->reset_notifiers, &now);

        }

        return now;

    case QEMU_CLOCK_VIRTUAL_RT:

        return cpu_get_clock();

    }

}
