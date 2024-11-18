static void monitor_command_cb(Monitor *mon, const char *cmdline, void *opaque)

{

    monitor_suspend(mon);

    handle_user_command(mon, cmdline);

    monitor_resume(mon);

}
