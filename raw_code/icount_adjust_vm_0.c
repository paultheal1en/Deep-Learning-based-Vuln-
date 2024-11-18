static void icount_adjust_vm(void *opaque)

{

    timer_mod(icount_vm_timer,

                   qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +

                   NANOSECONDS_PER_SECOND / 10);

    icount_adjust();

}
