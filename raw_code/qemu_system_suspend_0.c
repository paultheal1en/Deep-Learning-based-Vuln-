static void qemu_system_suspend(void)

{

    pause_all_vcpus();

    notifier_list_notify(&suspend_notifiers, NULL);

    runstate_set(RUN_STATE_SUSPENDED);

    monitor_protocol_event(QEVENT_SUSPEND, NULL);

    is_suspended = true;

}
