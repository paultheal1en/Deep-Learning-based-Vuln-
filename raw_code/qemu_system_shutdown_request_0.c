void qemu_system_shutdown_request(void)

{

    trace_qemu_system_shutdown_request();

    replay_shutdown_request();

    /* TODO - add a parameter to allow callers to specify reason */

    shutdown_requested = SHUTDOWN_CAUSE_HOST_ERROR;

    qemu_notify_event();

}
