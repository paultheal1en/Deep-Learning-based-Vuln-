static void uhci_async_cancel_all(UHCIState *s)

{

    UHCIQueue *queue, *nq;



    QTAILQ_FOREACH_SAFE(queue, &s->queues, next, nq) {

        uhci_queue_free(queue);

    }

}
