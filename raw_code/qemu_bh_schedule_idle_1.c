void qemu_bh_schedule_idle(QEMUBH *bh)

{

    if (bh->scheduled)

        return;

    bh->idle = 1;

    /* Make sure that idle & any writes needed by the callback are done

     * before the locations are read in the aio_bh_poll.

     */

    smp_wmb();

    bh->scheduled = 1;

}
