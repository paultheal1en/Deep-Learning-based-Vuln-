int qemu_calculate_timeout(void)

{

#ifndef CONFIG_IOTHREAD

    int timeout;



    if (!vm_running)

        timeout = 5000;

    else {

     /* XXX: use timeout computed from timers */

        int64_t add;

        int64_t delta;

        /* Advance virtual time to the next event.  */

	delta = qemu_icount_delta();

        if (delta > 0) {

            /* If virtual time is ahead of real time then just

               wait for IO.  */

            timeout = (delta + 999999) / 1000000;

        } else {

            /* Wait for either IO to occur or the next

               timer event.  */

            add = qemu_next_deadline();

            /* We advance the timer before checking for IO.

               Limit the amount we advance so that early IO

               activity won't get the guest too far ahead.  */

            if (add > 10000000)

                add = 10000000;

            delta += add;

            qemu_icount += qemu_icount_round (add);

            timeout = delta / 1000000;

            if (timeout < 0)

                timeout = 0;

        }

    }



    return timeout;

#else /* CONFIG_IOTHREAD */

    return 1000;

#endif

}
