void usb_packet_set_state(USBPacket *p, USBPacketState state)

{

    static const char *name[] = {

        [USB_PACKET_UNDEFINED] = "undef",

        [USB_PACKET_SETUP]     = "setup",

        [USB_PACKET_QUEUED]    = "queued",

        [USB_PACKET_ASYNC]     = "async",

        [USB_PACKET_COMPLETE]  = "complete",

        [USB_PACKET_CANCELED]  = "canceled",

    };

    USBDevice *dev = p->ep->dev;

    USBBus *bus = usb_bus_from_device(dev);



    trace_usb_packet_state_change(bus->busnr, dev->port->path, p->ep->nr,

                                  p, name[p->state], name[state]);

    p->state = state;

}
