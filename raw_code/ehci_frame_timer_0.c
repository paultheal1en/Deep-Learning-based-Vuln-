static void ehci_frame_timer(void *opaque)

{

    EHCIState *ehci = opaque;

    int schedules = 0;

    int64_t expire_time, t_now;

    uint64_t ns_elapsed;

    int frames;

    int i;

    int skipped_frames = 0;



    t_now = qemu_get_clock_ns(vm_clock);

    ns_elapsed = t_now - ehci->last_run_ns;

    frames = ns_elapsed / FRAME_TIMER_NS;



    if (ehci_periodic_enabled(ehci) || ehci->pstate != EST_INACTIVE) {

        schedules++;

        expire_time = t_now + (get_ticks_per_sec() / FRAME_TIMER_FREQ);



        for (i = 0; i < frames; i++) {

            ehci_update_frindex(ehci, 1);



            if (frames - i > ehci->maxframes) {

                skipped_frames++;

            } else {

                ehci_advance_periodic_state(ehci);

            }



            ehci->last_run_ns += FRAME_TIMER_NS;

        }

    } else {

        if (ehci->async_stepdown < ehci->maxframes / 2) {

            ehci->async_stepdown++;

        }

        expire_time = t_now + (get_ticks_per_sec()

                               * ehci->async_stepdown / FRAME_TIMER_FREQ);

        ehci_update_frindex(ehci, frames);

        ehci->last_run_ns += FRAME_TIMER_NS * frames;

    }



#if 0

    if (skipped_frames) {

        DPRINTF("WARNING - EHCI skipped %d frames\n", skipped_frames);

    }

#endif



    /*  Async is not inside loop since it executes everything it can once

     *  called

     */

    if (ehci_async_enabled(ehci) || ehci->astate != EST_INACTIVE) {

        schedules++;

        qemu_bh_schedule(ehci->async_bh);

    }



    if (schedules) {

        qemu_mod_timer(ehci->frame_timer, expire_time);

    }

}
