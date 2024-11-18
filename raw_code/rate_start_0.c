static void rate_start (SpiceRateCtl *rate)

{

    memset (rate, 0, sizeof (*rate));

    rate->start_ticks = qemu_get_clock (vm_clock);

}
