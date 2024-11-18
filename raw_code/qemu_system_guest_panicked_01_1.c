void qemu_system_guest_panicked(GuestPanicInformation *info)
{
    if (current_cpu) {
        current_cpu->crash_occurred = true;
    }
    qapi_event_send_guest_panicked(GUEST_PANIC_ACTION_PAUSE,
                                   !!info, info, &error_abort);
    vm_stop(RUN_STATE_GUEST_PANICKED);
    if (!no_shutdown) {
        qapi_event_send_guest_panicked(GUEST_PANIC_ACTION_POWEROFF,
                                       !!info, info, &error_abort);
        qemu_system_shutdown_request();
    }
    if (info) {
        if (info->type == GUEST_PANIC_INFORMATION_KIND_HYPER_V) {
            qemu_log_mask(LOG_GUEST_ERROR, "HV crash parameters: (%#"PRIx64
                          " %#"PRIx64" %#"PRIx64" %#"PRIx64" %#"PRIx64")\n",
                          info->u.hyper_v.data->arg1,
                          info->u.hyper_v.data->arg2,
                          info->u.hyper_v.data->arg3,
                          info->u.hyper_v.data->arg4,
                          info->u.hyper_v.data->arg5);
        }
        qapi_free_GuestPanicInformation(info);
    }
}