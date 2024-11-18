static void iscsi_retry_timer_expired(void *opaque)

{

    struct IscsiTask *iTask = opaque;

    iTask->complete = 1;

    if (iTask->co) {

        qemu_coroutine_enter(iTask->co, NULL);

    }

}
