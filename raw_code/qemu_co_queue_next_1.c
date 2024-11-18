bool qemu_co_queue_next(CoQueue *queue)

{

    struct unlock_bh *unlock_bh;

    Coroutine *next;



    next = QTAILQ_FIRST(&queue->entries);

    if (next) {

        QTAILQ_REMOVE(&queue->entries, next, co_queue_next);

        QTAILQ_INSERT_TAIL(&unlock_bh_queue, next, co_queue_next);

        trace_qemu_co_queue_next(next);



        unlock_bh = qemu_malloc(sizeof(*unlock_bh));

        unlock_bh->bh = qemu_bh_new(qemu_co_queue_next_bh, unlock_bh);

        qemu_bh_schedule(unlock_bh->bh);

    }



    return (next != NULL);

}
