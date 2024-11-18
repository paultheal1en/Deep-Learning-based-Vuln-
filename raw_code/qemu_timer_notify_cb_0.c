void qemu_timer_notify_cb(void *opaque, QEMUClockType type)

{

    qemu_notify_event();

}
