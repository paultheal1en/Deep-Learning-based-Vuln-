static int ehci_process_itd(EHCIState *ehci,

                            EHCIitd *itd)

{

    USBDevice *dev;

    USBEndpoint *ep;

    int ret;

    uint32_t i, len, pid, dir, devaddr, endp;

    uint32_t pg, off, ptr1, ptr2, max, mult;



    dir =(itd->bufptr[1] & ITD_BUFPTR_DIRECTION);

    devaddr = get_field(itd->bufptr[0], ITD_BUFPTR_DEVADDR);

    endp = get_field(itd->bufptr[0], ITD_BUFPTR_EP);

    max = get_field(itd->bufptr[1], ITD_BUFPTR_MAXPKT);

    mult = get_field(itd->bufptr[2], ITD_BUFPTR_MULT);



    for(i = 0; i < 8; i++) {

        if (itd->transact[i] & ITD_XACT_ACTIVE) {

            pg   = get_field(itd->transact[i], ITD_XACT_PGSEL);

            off  = itd->transact[i] & ITD_XACT_OFFSET_MASK;

            ptr1 = (itd->bufptr[pg] & ITD_BUFPTR_MASK);

            ptr2 = (itd->bufptr[pg+1] & ITD_BUFPTR_MASK);

            len  = get_field(itd->transact[i], ITD_XACT_LENGTH);



            if (len > max * mult) {

                len = max * mult;

            }



            if (len > BUFF_SIZE) {

                return USB_RET_PROCERR;

            }



            pci_dma_sglist_init(&ehci->isgl, &ehci->dev, 2);

            if (off + len > 4096) {

                /* transfer crosses page border */

                uint32_t len2 = off + len - 4096;

                uint32_t len1 = len - len2;

                qemu_sglist_add(&ehci->isgl, ptr1 + off, len1);

                qemu_sglist_add(&ehci->isgl, ptr2, len2);

            } else {

                qemu_sglist_add(&ehci->isgl, ptr1 + off, len);

            }



            pid = dir ? USB_TOKEN_IN : USB_TOKEN_OUT;



            dev = ehci_find_device(ehci, devaddr);

            ep = usb_ep_get(dev, pid, endp);

            usb_packet_setup(&ehci->ipacket, pid, ep);

            usb_packet_map(&ehci->ipacket, &ehci->isgl);



            ret = usb_handle_packet(dev, &ehci->ipacket);



            usb_packet_unmap(&ehci->ipacket);

            qemu_sglist_destroy(&ehci->isgl);



#if 0

            /*  In isoch, there is no facility to indicate a NAK so let's

             *  instead just complete a zero-byte transaction.  Setting

             *  DBERR seems too draconian.

             */



            if (ret == USB_RET_NAK) {

                if (ehci->isoch_pause > 0) {

                    DPRINTF("ISOCH: received a NAK but paused so returning\n");

                    ehci->isoch_pause--;

                    return 0;

                } else if (ehci->isoch_pause == -1) {

                    DPRINTF("ISOCH: recv NAK & isoch pause inactive, setting\n");

                    // Pause frindex for up to 50 msec waiting for data from

                    // remote

                    ehci->isoch_pause = 50;

                    return 0;

                } else {

                    DPRINTF("ISOCH: isoch pause timeout! return 0\n");

                    ret = 0;

                }

            } else {

                DPRINTF("ISOCH: received ACK, clearing pause\n");

                ehci->isoch_pause = -1;

            }

#else

            if (ret == USB_RET_NAK) {

                ret = 0;

            }

#endif



            if (ret >= 0) {

                if (!dir) {

                    /* OUT */

                    set_field(&itd->transact[i], len - ret, ITD_XACT_LENGTH);

                } else {

                    /* IN */

                    set_field(&itd->transact[i], ret, ITD_XACT_LENGTH);

                }



                if (itd->transact[i] & ITD_XACT_IOC) {

                    ehci_record_interrupt(ehci, USBSTS_INT);

                }

            }

            itd->transact[i] &= ~ITD_XACT_ACTIVE;

        }

    }

    return 0;

}
