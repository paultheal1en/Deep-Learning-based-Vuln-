static int ehci_state_execute(EHCIQueue *q)

{

    EHCIPacket *p = QTAILQ_FIRST(&q->packets);

    int again = 0;



    assert(p != NULL);

    assert(p->qtdaddr == q->qtdaddr);



    if (ehci_qh_do_overlay(q) != 0) {

        return -1;

    }



    // TODO verify enough time remains in the uframe as in 4.4.1.1

    // TODO write back ptr to async list when done or out of time

    // TODO Windows does not seem to ever set the MULT field



    if (!q->async) {

        int transactCtr = get_field(q->qh.epcap, QH_EPCAP_MULT);

        if (!transactCtr) {

            ehci_set_state(q->ehci, q->async, EST_HORIZONTALQH);

            again = 1;

            goto out;

        }

    }



    if (q->async) {

        ehci_set_usbsts(q->ehci, USBSTS_REC);

    }



    p->usb_status = ehci_execute(p, "process");

    if (p->usb_status == USB_RET_PROCERR) {

        again = -1;

        goto out;

    }

    if (p->usb_status == USB_RET_ASYNC) {

        ehci_flush_qh(q);

        trace_usb_ehci_packet_action(p->queue, p, "async");

        p->async = EHCI_ASYNC_INFLIGHT;

        ehci_set_state(q->ehci, q->async, EST_HORIZONTALQH);

        again = (ehci_fill_queue(p) == USB_RET_PROCERR) ? -1 : 1;

        goto out;

    }



    ehci_set_state(q->ehci, q->async, EST_EXECUTING);

    again = 1;



out:

    return again;

}
