static UHCIQueue *uhci_queue_get(UHCIState *s, UHCI_TD *td, USBEndpoint *ep)

{

    uint32_t token = uhci_queue_token(td);

    UHCIQueue *queue;



    QTAILQ_FOREACH(queue, &s->queues, next) {

        if (queue->token == token) {

            return queue;

        }

    }



    queue = g_new0(UHCIQueue, 1);

    queue->uhci = s;

    queue->token = token;

    queue->ep = ep;

    QTAILQ_INIT(&queue->asyncs);

    QTAILQ_INSERT_HEAD(&s->queues, queue, next);

    trace_usb_uhci_queue_add(queue->token);

    return queue;

}
