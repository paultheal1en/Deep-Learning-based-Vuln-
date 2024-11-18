static void usbredir_buffered_bulk_packet(void *priv, uint64_t id,

    struct usb_redir_buffered_bulk_packet_header *buffered_bulk_packet,

    uint8_t *data, int data_len)

{

    USBRedirDevice *dev = priv;

    uint8_t status, ep = buffered_bulk_packet->endpoint;

    void *free_on_destroy;

    int i, len;



    DPRINTF("buffered-bulk-in status %d ep %02X len %d id %"PRIu64"\n",

            buffered_bulk_packet->status, ep, data_len, id);



    if (dev->endpoint[EP2I(ep)].type != USB_ENDPOINT_XFER_BULK) {

        ERROR("received buffered-bulk packet for non bulk ep %02X\n", ep);

        free(data);

        return;

    }



    if (dev->endpoint[EP2I(ep)].bulk_receiving_started == 0) {

        DPRINTF("received buffered-bulk packet on not started ep %02X\n", ep);

        free(data);

        return;

    }



    /* Data must be in maxp chunks for buffered_bulk_add_*_data_to_packet */

    len = dev->endpoint[EP2I(ep)].max_packet_size;

    status = usb_redir_success;

    free_on_destroy = NULL;

    for (i = 0; i < data_len; i += len) {

        if (len >= (data_len - i)) {

            len = data_len - i;

            status = buffered_bulk_packet->status;

            free_on_destroy = data;

        }

        /* bufp_alloc also adds the packet to the ep queue */

        bufp_alloc(dev, data + i, len, status, ep, free_on_destroy);

    }



    if (dev->endpoint[EP2I(ep)].pending_async_packet) {

        USBPacket *p = dev->endpoint[EP2I(ep)].pending_async_packet;

        dev->endpoint[EP2I(ep)].pending_async_packet = NULL;

        usbredir_buffered_bulk_in_complete(dev, p, ep);

        usb_packet_complete(&dev->dev, p);

    }

}
