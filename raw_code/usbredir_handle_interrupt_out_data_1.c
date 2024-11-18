static void usbredir_handle_interrupt_out_data(USBRedirDevice *dev,

                                               USBPacket *p, uint8_t ep)

{

    /* Output interrupt endpoint, normal async operation */

    struct usb_redir_interrupt_packet_header interrupt_packet;

    uint8_t buf[p->iov.size];



    DPRINTF("interrupt-out ep %02X len %zd id %"PRIu64"\n", ep,

            p->iov.size, p->id);



    if (usbredir_already_in_flight(dev, p->id)) {

        p->status = USB_RET_ASYNC;

        return;

    }



    interrupt_packet.endpoint  = ep;

    interrupt_packet.length    = p->iov.size;



    usb_packet_copy(p, buf, p->iov.size);

    usbredir_log_data(dev, "interrupt data out:", buf, p->iov.size);

    usbredirparser_send_interrupt_packet(dev->parser, p->id,

                                    &interrupt_packet, buf, p->iov.size);

    usbredirparser_do_write(dev->parser);

    p->status = USB_RET_ASYNC;

}
