void bdrv_error_action(BlockDriverState *bs, BlockErrorAction action,

                       bool is_read, int error)

{

    assert(error >= 0);



    if (action == BLOCK_ERROR_ACTION_STOP) {

        /* First set the iostatus, so that "info block" returns an iostatus

         * that matches the events raised so far (an additional error iostatus

         * is fine, but not a lost one).

         */

        bdrv_iostatus_set_err(bs, error);



        /* Then raise the request to stop the VM and the event.

         * qemu_system_vmstop_request_prepare has two effects.  First,

         * it ensures that the STOP event always comes after the

         * BLOCK_IO_ERROR event.  Second, it ensures that even if management

         * can observe the STOP event and do a "cont" before the STOP

         * event is issued, the VM will not stop.  In this case, vm_start()

         * also ensures that the STOP/RESUME pair of events is emitted.

         */

        qemu_system_vmstop_request_prepare();

        send_qmp_error_event(bs, action, is_read, error);

        qemu_system_vmstop_request(RUN_STATE_IO_ERROR);

    } else {

        send_qmp_error_event(bs, action, is_read, error);

    }

}
