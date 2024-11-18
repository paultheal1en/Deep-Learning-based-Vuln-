static void iscsi_attach_aio_context(BlockDriverState *bs,

                                     AioContext *new_context)

{

    IscsiLun *iscsilun = bs->opaque;



    iscsilun->aio_context = new_context;

    iscsi_set_events(iscsilun);



#if defined(LIBISCSI_FEATURE_NOP_COUNTER)

    /* Set up a timer for sending out iSCSI NOPs */

    iscsilun->nop_timer = aio_timer_new(iscsilun->aio_context,

                                        QEMU_CLOCK_REALTIME, SCALE_MS,

                                        iscsi_nop_timed_event, iscsilun);

    timer_mod(iscsilun->nop_timer,

              qemu_clock_get_ms(QEMU_CLOCK_REALTIME) + NOP_INTERVAL);

#endif

}
