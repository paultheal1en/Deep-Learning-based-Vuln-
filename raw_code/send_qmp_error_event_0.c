static void send_qmp_error_event(BlockDriverState *bs,

                                 BlockErrorAction action,

                                 bool is_read, int error)

{

    IoOperationType optype;



    optype = is_read ? IO_OPERATION_TYPE_READ : IO_OPERATION_TYPE_WRITE;

    qapi_event_send_block_io_error(bdrv_get_device_name(bs), optype, action,

                                   bdrv_iostatus_is_enabled(bs),

                                   error == ENOSPC, strerror(error),

                                   &error_abort);

}
