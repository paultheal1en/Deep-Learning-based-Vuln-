void qemu_clock_unregister_reset_notifier(QEMUClockType type,

                                          Notifier *notifier)

{

    notifier_remove(notifier);

}
