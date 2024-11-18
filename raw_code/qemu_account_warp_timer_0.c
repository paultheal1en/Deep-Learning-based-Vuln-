static void qemu_account_warp_timer(void)

{

    if (!use_icount || !icount_sleep) {

        return;

    }



    /* Nothing to do if the VM is stopped: QEMU_CLOCK_VIRTUAL timers

     * do not fire, so computing the deadline does not make sense.

     */

    if (!runstate_is_running()) {

        return;

    }



    /* warp clock deterministically in record/replay mode */

    if (!replay_checkpoint(CHECKPOINT_CLOCK_WARP_ACCOUNT)) {

        return;

    }



    timer_del(icount_warp_timer);

    icount_warp_rt();

}
