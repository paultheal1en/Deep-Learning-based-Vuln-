static int usbredir_handle_bulk_data(USBRedirDevice *dev, USBPacket *p,

                                      uint8_t ep)

{

    AsyncURB *aurb = async_alloc(dev, p);

    struct usb_redir_bulk_packet_header bulk_packet;



    DPRINTF("bulk-out ep %02X len %zd id %u\n", ep,

            p->iov.size, aurb->packet_id);



    bulk_packet.endpoint  = ep;

    bulk_packet.length    = p->iov.size;

    bulk_packet.stream_id = 0;

    aurb->bulk_packet = bulk_packet;



    if (ep & USB_DIR_IN) {

        usbredirparser_send_bulk_packet(dev->parser, aurb->packet_id,

                                        &bulk_packet, NULL, 0);

    } else {

        uint8_t buf[p->iov.size];

        usb_packet_copy(p, buf, p->iov.size);

        usbredir_log_data(dev, "bulk data out:", buf, p->iov.size);

        usbredirparser_send_bulk_packet(dev->parser, aurb->packet_id,

                                        &bulk_packet, buf, p->iov.size);

    }

    usbredirparser_do_write(dev->parser);

    return USB_RET_ASYNC;

}
