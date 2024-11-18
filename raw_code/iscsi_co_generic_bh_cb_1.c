static void iscsi_co_generic_bh_cb(void *opaque)

{

    struct IscsiTask *iTask = opaque;

    iTask->complete = 1;

    qemu_bh_delete(iTask->bh);

    qemu_coroutine_enter(iTask->co, NULL);

}
