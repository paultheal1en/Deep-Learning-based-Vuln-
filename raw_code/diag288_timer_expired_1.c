static void diag288_timer_expired(void *dev)

{

    qemu_log_mask(CPU_LOG_RESET, "Watchdog timer expired.\n");

    watchdog_perform_action();

    /* Reset the watchdog only if the guest was notified about expiry. */

    switch (get_watchdog_action()) {

    case WDT_DEBUG:

    case WDT_NONE:

    case WDT_PAUSE:

         return;

    }

    wdt_diag288_reset(dev);

}
