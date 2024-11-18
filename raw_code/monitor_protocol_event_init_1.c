static void monitor_protocol_event_init(void)

{

    qemu_mutex_init(&monitor_event_state_lock);

    /* Limit RTC & BALLOON events to 1 per second */

    monitor_protocol_event_throttle(QEVENT_RTC_CHANGE, 1000);

    monitor_protocol_event_throttle(QEVENT_BALLOON_CHANGE, 1000);

    monitor_protocol_event_throttle(QEVENT_WATCHDOG, 1000);

}
