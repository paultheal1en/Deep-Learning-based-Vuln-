void qemu_system_suspend_request(void)

{

    if (is_suspended) {

        return;

    }

    suspend_requested = 1;

    cpu_stop_current();

    qemu_notify_event();

}
