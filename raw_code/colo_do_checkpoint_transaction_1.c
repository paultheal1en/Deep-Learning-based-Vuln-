static int colo_do_checkpoint_transaction(MigrationState *s,

                                          QIOChannelBuffer *bioc,

                                          QEMUFile *fb)

{

    Error *local_err = NULL;

    int ret = -1;



    colo_send_message(s->to_dst_file, COLO_MESSAGE_CHECKPOINT_REQUEST,

                      &local_err);

    if (local_err) {





    colo_receive_check_message(s->rp_state.from_dst_file,

                    COLO_MESSAGE_CHECKPOINT_REPLY, &local_err);

    if (local_err) {



    /* Reset channel-buffer directly */

    qio_channel_io_seek(QIO_CHANNEL(bioc), 0, 0, NULL);

    bioc->usage = 0;



    qemu_mutex_lock_iothread();





    vm_stop_force_state(RUN_STATE_COLO);


    trace_colo_vm_state_change("run", "stop");










    /* Disable block migration */

    s->params.blk = 0;

    s->params.shared = 0;

    qemu_savevm_state_header(fb);

    qemu_savevm_state_begin(fb, &s->params);

    qemu_mutex_lock_iothread();

    qemu_savevm_state_complete_precopy(fb, false);




    qemu_fflush(fb);



    colo_send_message(s->to_dst_file, COLO_MESSAGE_VMSTATE_SEND, &local_err);

    if (local_err) {




     * We need the size of the VMstate data in Secondary side,

     * With which we can decide how much data should be read.


    colo_send_message_value(s->to_dst_file, COLO_MESSAGE_VMSTATE_SIZE,

                            bioc->usage, &local_err);

    if (local_err) {





    qemu_put_buffer(s->to_dst_file, bioc->data, bioc->usage);

    qemu_fflush(s->to_dst_file);

    ret = qemu_file_get_error(s->to_dst_file);

    if (ret < 0) {





    colo_receive_check_message(s->rp_state.from_dst_file,

                       COLO_MESSAGE_VMSTATE_RECEIVED, &local_err);

    if (local_err) {





    colo_receive_check_message(s->rp_state.from_dst_file,

                       COLO_MESSAGE_VMSTATE_LOADED, &local_err);

    if (local_err) {





    ret = 0;



    qemu_mutex_lock_iothread();

    vm_start();


    trace_colo_vm_state_change("stop", "run");



out:

    if (local_err) {

        error_report_err(local_err);


    return ret;
