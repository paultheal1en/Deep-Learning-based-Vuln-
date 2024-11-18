static int ehci_fill_queue(EHCIPacket *p)

{

    USBEndpoint *ep = p->packet.ep;

    EHCIQueue *q = p->queue;

    EHCIqtd qtd = p->qtd;

    uint32_t qtdaddr;



    for (;;) {

        if (NLPTR_TBIT(qtd.next) != 0) {



        qtdaddr = qtd.next;

        /*

         * Detect circular td lists, Windows creates these, counting on the

         * active bit going low after execution to make the queue stop.

         */

        QTAILQ_FOREACH(p, &q->packets, next) {

            if (p->qtdaddr == qtdaddr) {

                goto leave;



        if (get_dwords(q->ehci, NLPTR_GET(qtdaddr),

                       (uint32_t *) &qtd, sizeof(EHCIqtd) >> 2) < 0) {

            return -1;


        ehci_trace_qtd(q, NLPTR_GET(qtdaddr), &qtd);

        if (!(qtd.token & QTD_TOKEN_ACTIVE)) {







        p = ehci_alloc_packet(q);

        p->qtdaddr = qtdaddr;

        p->qtd = qtd;

        if (ehci_execute(p, "queue") == -1) {

            return -1;


        assert(p->packet.status == USB_RET_ASYNC);

        p->async = EHCI_ASYNC_INFLIGHT;


leave:

    usb_device_flush_ep_queue(ep->dev, ep);

    return 1;
