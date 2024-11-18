void usb_packet_unmap(USBPacket *p)

{

    int is_write = (p->pid == USB_TOKEN_IN);

    int i;



    for (i = 0; i < p->iov.niov; i++) {

        cpu_physical_memory_unmap(p->iov.iov[i].iov_base,

                                  p->iov.iov[i].iov_len, is_write,

                                  p->iov.iov[i].iov_len);

    }

}
