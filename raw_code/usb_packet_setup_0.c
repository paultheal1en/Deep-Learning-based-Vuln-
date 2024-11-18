void usb_packet_setup(USBPacket *p, int pid, uint8_t addr, uint8_t ep)

{

    assert(!usb_packet_is_inflight(p));

    p->state = USB_PACKET_SETUP;

    p->pid = pid;

    p->devaddr = addr;

    p->devep = ep;

    p->result = 0;

    qemu_iovec_reset(&p->iov);

}
