static bool qemu_co_queue_do_restart(CoQueue *queue, bool single)

{

    Coroutine *next;

    CoQueueNextData *data;



    if (QTAILQ_EMPTY(&queue->entries)) {

        return false;

    }



    data = g_slice_new(CoQueueNextData);

    data->bh = aio_bh_new(queue->ctx, qemu_co_queue_next_bh, data);

    QTAILQ_INIT(&data->entries);

    qemu_bh_schedule(data->bh);



    while ((next = QTAILQ_FIRST(&queue->entries)) != NULL) {

        QTAILQ_REMOVE(&queue->entries, next, co_queue_next);

        QTAILQ_INSERT_TAIL(&data->entries, next, co_queue_next);

        trace_qemu_co_queue_next(next);

        if (single) {

            break;

        }

    }

    return true;

}
