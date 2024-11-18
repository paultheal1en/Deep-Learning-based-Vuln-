static void uhci_queue_free(UHCIQueue *queue)

{

    UHCIState *s = queue->uhci;

    UHCIAsync *async;



    while (!QTAILQ_EMPTY(&queue->asyncs)) {

        async = QTAILQ_FIRST(&queue->asyncs);

        uhci_async_cancel(async);

    }



    trace_usb_uhci_queue_del(queue->token);

    QTAILQ_REMOVE(&s->queues, queue, next);

    g_free(queue);

}
