static int xhci_setup_packet(XHCITransfer *xfer, XHCIPort *port, int ep)

{

    usb_packet_setup(&xfer->packet,

                     xfer->in_xfer ? USB_TOKEN_IN : USB_TOKEN_OUT,

                     xfer->xhci->slots[xfer->slotid-1].devaddr,

                     ep & 0x7f);

    usb_packet_addbuf(&xfer->packet, xfer->data, xfer->data_length);

    DPRINTF("xhci: setup packet pid 0x%x addr %d ep %d\n",

            xfer->packet.pid, xfer->packet.devaddr, xfer->packet.devep);

    return 0;

}
