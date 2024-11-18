void qemu_clock_register_reset_notifier(QEMUClockType type,

                                        Notifier *notifier)

{

    QEMUClock *clock = qemu_clock_ptr(type);

    notifier_list_add(&clock->reset_notifiers, notifier);

}
