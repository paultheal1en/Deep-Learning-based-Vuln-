bool qemu_co_queue_next(CoQueue *queue)

{

    Coroutine *next;



    next = QTAILQ_FIRST(&queue->entries);

    if (next) {

        QTAILQ_REMOVE(&queue->entries, next, co_queue_next);

        QTAILQ_INSERT_TAIL(&unlock_bh_queue, next, co_queue_next);

        trace_qemu_co_queue_next(next);

        qemu_bh_schedule(unlock_bh);

    }



    return (next != NULL);

}
