void configure_icount(QemuOpts *opts, Error **errp)

{

    const char *option;

    char *rem_str = NULL;



    option = qemu_opt_get(opts, "shift");

    if (!option) {

        if (qemu_opt_get(opts, "align") != NULL) {

            error_setg(errp, "Please specify shift option when using align");

        }

        return;

    }



    icount_sleep = qemu_opt_get_bool(opts, "sleep", true);

    if (icount_sleep) {

        icount_warp_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL_RT,

                                         icount_dummy_timer, NULL);

    }



    icount_align_option = qemu_opt_get_bool(opts, "align", false);



    if (icount_align_option && !icount_sleep) {

        error_setg(errp, "align=on and sleep=off are incompatible");

    }

    if (strcmp(option, "auto") != 0) {

        errno = 0;

        icount_time_shift = strtol(option, &rem_str, 0);

        if (errno != 0 || *rem_str != '\0' || !strlen(option)) {

            error_setg(errp, "icount: Invalid shift value");

        }

        use_icount = 1;

        return;

    } else if (icount_align_option) {

        error_setg(errp, "shift=auto and align=on are incompatible");

    } else if (!icount_sleep) {

        error_setg(errp, "shift=auto and sleep=off are incompatible");

    }



    use_icount = 2;



    /* 125MIPS seems a reasonable initial guess at the guest speed.

       It will be corrected fairly quickly anyway.  */

    icount_time_shift = 3;



    /* Have both realtime and virtual time triggers for speed adjustment.

       The realtime trigger catches emulated time passing too slowly,

       the virtual time trigger catches emulated time passing too fast.

       Realtime triggers occur even when idle, so use them less frequently

       than VM triggers.  */

    icount_rt_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL_RT,

                                   icount_adjust_rt, NULL);

    timer_mod(icount_rt_timer,

                   qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL_RT) + 1000);

    icount_vm_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,

                                        icount_adjust_vm, NULL);

    timer_mod(icount_vm_timer,

                   qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +

                   get_ticks_per_sec() / 10);

}
