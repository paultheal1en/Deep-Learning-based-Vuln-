void watchdog_perform_action(void)

{

    switch (watchdog_action) {

    case WDT_RESET:             /* same as 'system_reset' in monitor */

        qapi_event_send_watchdog(WATCHDOG_EXPIRATION_ACTION_RESET, &error_abort);

        qemu_system_reset_request();

        break;



    case WDT_SHUTDOWN:          /* same as 'system_powerdown' in monitor */

        qapi_event_send_watchdog(WATCHDOG_EXPIRATION_ACTION_SHUTDOWN, &error_abort);

        qemu_system_powerdown_request();

        break;



    case WDT_POWEROFF:          /* same as 'quit' command in monitor */

        qapi_event_send_watchdog(WATCHDOG_EXPIRATION_ACTION_POWEROFF, &error_abort);

        exit(0);



    case WDT_PAUSE:             /* same as 'stop' command in monitor */

        qapi_event_send_watchdog(WATCHDOG_EXPIRATION_ACTION_PAUSE, &error_abort);

        vm_stop(RUN_STATE_WATCHDOG);

        break;



    case WDT_DEBUG:

        qapi_event_send_watchdog(WATCHDOG_EXPIRATION_ACTION_DEBUG, &error_abort);

        fprintf(stderr, "watchdog: timer fired\n");

        break;



    case WDT_NONE:

        qapi_event_send_watchdog(WATCHDOG_EXPIRATION_ACTION_NONE, &error_abort);

        break;

    }

}
