static int usbredir_handle_control(USBDevice *udev, USBPacket *p,

        int request, int value, int index, int length, uint8_t *data)

{

    USBRedirDevice *dev = DO_UPCAST(USBRedirDevice, dev, udev);

    struct usb_redir_control_packet_header control_packet;

    AsyncURB *aurb;



    /* Special cases for certain standard device requests */

    switch (request) {

    case DeviceOutRequest | USB_REQ_SET_ADDRESS:

        DPRINTF("set address %d\n", value);

        dev->dev.addr = value;

        return 0;

    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

        return usbredir_set_config(dev, p, value & 0xff);

    case DeviceRequest | USB_REQ_GET_CONFIGURATION:

        return usbredir_get_config(dev, p);

    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        return usbredir_set_interface(dev, p, index, value);

    case InterfaceRequest | USB_REQ_GET_INTERFACE:

        return usbredir_get_interface(dev, p, index);

    }



    /* "Normal" ctrl requests */

    aurb = async_alloc(dev, p);



    /* Note request is (bRequestType << 8) | bRequest */

    DPRINTF("ctrl-out type 0x%x req 0x%x val 0x%x index %d len %d id %u\n",

            request >> 8, request & 0xff, value, index, length,

            aurb->packet_id);



    control_packet.request     = request & 0xFF;

    control_packet.requesttype = request >> 8;

    control_packet.endpoint    = control_packet.requesttype & USB_DIR_IN;

    control_packet.value       = value;

    control_packet.index       = index;

    control_packet.length      = length;

    aurb->control_packet       = control_packet;



    if (control_packet.requesttype & USB_DIR_IN) {

        usbredirparser_send_control_packet(dev->parser, aurb->packet_id,

                                           &control_packet, NULL, 0);

    } else {

        usbredir_log_data(dev, "ctrl data out:", data, length);

        usbredirparser_send_control_packet(dev->parser, aurb->packet_id,

                                           &control_packet, data, length);

    }

    usbredirparser_do_write(dev->parser);

    return USB_RET_ASYNC;

}
