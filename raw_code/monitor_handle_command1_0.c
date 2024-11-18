static void monitor_handle_command1(void *opaque, const char *cmdline)

{

    monitor_handle_command(cmdline);

    if (!monitor_suspended)

        monitor_start_input();

    else

        monitor_suspended = 2;

}
