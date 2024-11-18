void qemu_co_queue_init(CoQueue *queue)

{

    QTAILQ_INIT(&queue->entries);



    /* This will be exposed to callers once there are multiple AioContexts */

    queue->ctx = qemu_get_aio_context();

}
