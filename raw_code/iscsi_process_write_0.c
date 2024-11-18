iscsi_process_write(void *arg)

{

    IscsiLun *iscsilun = arg;

    struct iscsi_context *iscsi = iscsilun->iscsi;



    aio_context_acquire(iscsilun->aio_context);

    iscsi_service(iscsi, POLLOUT);

    iscsi_set_events(iscsilun);

    aio_context_release(iscsilun->aio_context);

}
