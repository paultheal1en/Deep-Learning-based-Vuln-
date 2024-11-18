static void *iothread_run(void *opaque)

{

    IOThread *iothread = opaque;



    qemu_mutex_lock(&iothread->init_done_lock);

    iothread->thread_id = qemu_get_thread_id();

    qemu_cond_signal(&iothread->init_done_cond);

    qemu_mutex_unlock(&iothread->init_done_lock);



    while (!iothread->stopping) {

        aio_context_acquire(iothread->ctx);

        while (!iothread->stopping && aio_poll(iothread->ctx, true)) {

            /* Progress was made, keep going */

        }

        aio_context_release(iothread->ctx);

    }

    return NULL;

}
