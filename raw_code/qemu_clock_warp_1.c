void qemu_clock_warp(QEMUClockType type)

{

    int64_t deadline;



    /*

     * There are too many global variables to make the "warp" behavior

     * applicable to other clocks.  But a clock argument removes the

     * need for if statements all over the place.

     */

    if (type != QEMU_CLOCK_VIRTUAL || !use_icount) {

        return;

    }



    /*

     * If the CPUs have been sleeping, advance QEMU_CLOCK_VIRTUAL timer now.

     * This ensures that the deadline for the timer is computed correctly below.

     * This also makes sure that the insn counter is synchronized before the

     * CPU starts running, in case the CPU is woken by an event other than

     * the earliest QEMU_CLOCK_VIRTUAL timer.

     */

    icount_warp_rt(NULL);

    if (!all_cpu_threads_idle() || !qemu_clock_has_timers(QEMU_CLOCK_VIRTUAL)) {

        timer_del(icount_warp_timer);

        return;

    }



    if (qtest_enabled()) {

        /* When testing, qtest commands advance icount.  */

	return;

    }



    vm_clock_warp_start = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

    /* We want to use the earliest deadline from ALL vm_clocks */

    deadline = qemu_clock_deadline_ns_all(QEMU_CLOCK_VIRTUAL);



    /* Maintain prior (possibly buggy) behaviour where if no deadline

     * was set (as there is no QEMU_CLOCK_VIRTUAL timer) or it is more than

     * INT32_MAX nanoseconds ahead, we still use INT32_MAX

     * nanoseconds.

     */

    if ((deadline < 0) || (deadline > INT32_MAX)) {

        deadline = INT32_MAX;

    }



    if (deadline > 0) {

        /*

         * Ensure QEMU_CLOCK_VIRTUAL proceeds even when the virtual CPU goes to

         * sleep.  Otherwise, the CPU might be waiting for a future timer

         * interrupt to wake it up, but the interrupt never comes because

         * the vCPU isn't running any insns and thus doesn't advance the

         * QEMU_CLOCK_VIRTUAL.

         *

         * An extreme solution for this problem would be to never let VCPUs

         * sleep in icount mode if there is a pending QEMU_CLOCK_VIRTUAL

         * timer; rather time could just advance to the next QEMU_CLOCK_VIRTUAL

         * event.  Instead, we do stop VCPUs and only advance QEMU_CLOCK_VIRTUAL

         * after some e"real" time, (related to the time left until the next

         * event) has passed. The QEMU_CLOCK_REALTIME timer will do this.

         * This avoids that the warps are visible externally; for example,

         * you will not be sending network packets continuously instead of

         * every 100ms.

         */

        timer_mod(icount_warp_timer, vm_clock_warp_start + deadline);

    } else if (deadline == 0) {

        qemu_clock_notify(QEMU_CLOCK_VIRTUAL);

    }

}
