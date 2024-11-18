void wdt_ib700_init(void)

{

    watchdog_add_model(&model);

    timer = qemu_new_timer(vm_clock, ib700_timer_expired, NULL);

}
