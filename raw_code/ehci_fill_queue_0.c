static int ehci_fill_queue(EHCIPacket *p)

{

    EHCIQueue *q = p->queue;

    EHCIqtd qtd = p->qtd;

    uint32_t qtdaddr;



    for (;;) {

        if (NLPTR_TBIT(qtd.altnext) == 0) {

            break;

        }

        if (NLPTR_TBIT(qtd.next) != 0) {

            break;

        }

        qtdaddr = qtd.next;

        get_dwords(q->ehci, NLPTR_GET(qtdaddr),

                   (uint32_t *) &qtd, sizeof(EHCIqtd) >> 2);

        ehci_trace_qtd(q, NLPTR_GET(qtdaddr), &qtd);

        if (!(qtd.token & QTD_TOKEN_ACTIVE)) {

            break;

        }

        p = ehci_alloc_packet(q);

        p->qtdaddr = qtdaddr;

        p->qtd = qtd;

        p->usb_status = ehci_execute(p, "queue");

        if (p->usb_status == USB_RET_PROCERR) {

            break;

        }

        assert(p->usb_status == USB_RET_ASYNC);

        p->async = EHCI_ASYNC_INFLIGHT;

    }

    return p->usb_status;

}
