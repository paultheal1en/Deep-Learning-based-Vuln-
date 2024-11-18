void usb_packet_complete(USBDevice *dev, USBPacket *p)

{

    USBEndpoint *ep = p->ep;

    int ret;



    assert(p->state == USB_PACKET_ASYNC);

    assert(QTAILQ_FIRST(&ep->queue) == p);

    usb_packet_set_state(p, USB_PACKET_COMPLETE);

    QTAILQ_REMOVE(&ep->queue, p, queue);

    dev->port->ops->complete(dev->port, p);



    while (!QTAILQ_EMPTY(&ep->queue)) {

        p = QTAILQ_FIRST(&ep->queue);

        if (p->state == USB_PACKET_ASYNC) {

            break;

        }

        assert(p->state == USB_PACKET_QUEUED);

        ret = usb_process_one(p);

        if (ret == USB_RET_ASYNC) {

            usb_packet_set_state(p, USB_PACKET_ASYNC);

            break;

        }

        p->result = ret;

        usb_packet_set_state(p, USB_PACKET_COMPLETE);

        QTAILQ_REMOVE(&ep->queue, p, queue);

        dev->port->ops->complete(dev->port, p);

    }

}
