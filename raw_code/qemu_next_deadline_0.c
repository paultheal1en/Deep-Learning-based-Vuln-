static int64_t qemu_next_deadline(void)

{

    int64_t delta;



    if (active_timers[QEMU_CLOCK_VIRTUAL]) {

        delta = active_timers[QEMU_CLOCK_VIRTUAL]->expire_time -

                     qemu_get_clock(vm_clock);

    } else {

        /* To avoid problems with overflow limit this to 2^32.  */

        delta = INT32_MAX;

    }



    if (delta < 0)

        delta = 0;



    return delta;

}
