int usb_packet_map(USBPacket *p, QEMUSGList *sgl)

{

    int is_write = (p->pid == USB_TOKEN_IN);

    target_phys_addr_t len;

    void *mem;

    int i;



    for (i = 0; i < sgl->nsg; i++) {

        len = sgl->sg[i].len;

        mem = cpu_physical_memory_map(sgl->sg[i].base, &len,

                                      is_write);

        if (!mem) {

            goto err;

        }

        qemu_iovec_add(&p->iov, mem, len);

        if (len != sgl->sg[i].len) {

            goto err;

        }

    }

    return 0;



err:

    usb_packet_unmap(p);

    return -1;

}
