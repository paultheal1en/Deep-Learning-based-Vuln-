static int uhci_handle_td(UHCIState *s, UHCIQueue *q,

                          UHCI_TD *td, uint32_t td_addr, uint32_t *int_mask)

{

    UHCIAsync *async;

    int len = 0, max_len;

    bool spd;

    bool queuing = (q != NULL);

    uint8_t pid = td->token & 0xff;



    /* Is active ? */

    if (!(td->ctrl & TD_CTRL_ACTIVE)) {

        /*

         * ehci11d spec page 22: "Even if the Active bit in the TD is already

         * cleared when the TD is fetched ... an IOC interrupt is generated"

         */

        if (td->ctrl & TD_CTRL_IOC) {

                *int_mask |= 0x01;

        }

        return TD_RESULT_NEXT_QH;

    }



    async = uhci_async_find_td(s, td_addr, td);

    if (async) {

        /* Already submitted */

        async->queue->valid = 32;



        if (!async->done)

            return TD_RESULT_ASYNC_CONT;

        if (queuing) {

            /* we are busy filling the queue, we are not prepared

               to consume completed packages then, just leave them

               in async state */

            return TD_RESULT_ASYNC_CONT;

        }



        uhci_async_unlink(async);

        goto done;

    }



    /* Allocate new packet */

    if (q == NULL) {

        USBDevice *dev = uhci_find_device(s, (td->token >> 8) & 0x7f);

        USBEndpoint *ep = usb_ep_get(dev, pid, (td->token >> 15) & 0xf);

        q = uhci_queue_get(s, td, ep);

    }

    async = uhci_async_alloc(q, td_addr);



    /* valid needs to be large enough to handle 10 frame delay

     * for initial isochronous requests

     */

    async->queue->valid = 32;



    max_len = ((td->token >> 21) + 1) & 0x7ff;

    spd = (pid == USB_TOKEN_IN && (td->ctrl & TD_CTRL_SPD) != 0);

    usb_packet_setup(&async->packet, pid, q->ep, td_addr, spd,

                     (td->ctrl & TD_CTRL_IOC) != 0);

    qemu_sglist_add(&async->sgl, td->buffer, max_len);

    usb_packet_map(&async->packet, &async->sgl);



    switch(pid) {

    case USB_TOKEN_OUT:

    case USB_TOKEN_SETUP:

        len = usb_handle_packet(q->ep->dev, &async->packet);

        if (len >= 0)

            len = max_len;

        break;



    case USB_TOKEN_IN:

        len = usb_handle_packet(q->ep->dev, &async->packet);

        break;



    default:

        /* invalid pid : frame interrupted */

        usb_packet_unmap(&async->packet, &async->sgl);

        uhci_async_free(async);

        s->status |= UHCI_STS_HCPERR;

        uhci_update_irq(s);

        return TD_RESULT_STOP_FRAME;

    }

 

    if (len == USB_RET_ASYNC) {

        uhci_async_link(async);

        if (!queuing) {

            uhci_queue_fill(q, td);

        }

        return TD_RESULT_ASYNC_START;

    }



    async->packet.result = len;



done:

    len = uhci_complete_td(s, td, async, int_mask);

    usb_packet_unmap(&async->packet, &async->sgl);

    uhci_async_free(async);

    return len;

}
