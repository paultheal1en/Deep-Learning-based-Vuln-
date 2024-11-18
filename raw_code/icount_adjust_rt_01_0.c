static void icount_adjust_rt(void *opaque)

{

    timer_mod(icount_rt_timer,

              qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL_RT) + 1000);

    icount_adjust();

}
