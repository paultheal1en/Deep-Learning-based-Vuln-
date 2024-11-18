static int ohci_service_iso_td(OHCIState *ohci, struct ohci_ed *ed,

                               int completion)

{

    int dir;

    size_t len = 0;

#ifdef DEBUG_ISOCH

    const char *str = NULL;

#endif

    int pid;

    int ret;

    int i;

    USBDevice *dev;

    struct ohci_iso_td iso_td;

    uint32_t addr;

    uint16_t starting_frame;

    int16_t relative_frame_number;

    int frame_count;

    uint32_t start_offset, next_offset, end_offset = 0;

    uint32_t start_addr, end_addr;



    addr = ed->head & OHCI_DPTR_MASK;



    if (!ohci_read_iso_td(ohci, addr, &iso_td)) {

        printf("usb-ohci: ISO_TD read error at %x\n", addr);

        return 0;

    }



    starting_frame = OHCI_BM(iso_td.flags, TD_SF);

    frame_count = OHCI_BM(iso_td.flags, TD_FC);

    relative_frame_number = USUB(ohci->frame_number, starting_frame); 



#ifdef DEBUG_ISOCH

    printf("--- ISO_TD ED head 0x%.8x tailp 0x%.8x\n"

           "0x%.8x 0x%.8x 0x%.8x 0x%.8x\n"

           "0x%.8x 0x%.8x 0x%.8x 0x%.8x\n"

           "0x%.8x 0x%.8x 0x%.8x 0x%.8x\n"

           "frame_number 0x%.8x starting_frame 0x%.8x\n"

           "frame_count  0x%.8x relative %d\n"

           "di 0x%.8x cc 0x%.8x\n",

           ed->head & OHCI_DPTR_MASK, ed->tail & OHCI_DPTR_MASK,

           iso_td.flags, iso_td.bp, iso_td.next, iso_td.be,

           iso_td.offset[0], iso_td.offset[1], iso_td.offset[2], iso_td.offset[3],

           iso_td.offset[4], iso_td.offset[5], iso_td.offset[6], iso_td.offset[7],

           ohci->frame_number, starting_frame, 

           frame_count, relative_frame_number,         

           OHCI_BM(iso_td.flags, TD_DI), OHCI_BM(iso_td.flags, TD_CC));

#endif



    if (relative_frame_number < 0) {

        DPRINTF("usb-ohci: ISO_TD R=%d < 0\n", relative_frame_number);

        return 1;

    } else if (relative_frame_number > frame_count) {

        /* ISO TD expired - retire the TD to the Done Queue and continue with

           the next ISO TD of the same ED */

        DPRINTF("usb-ohci: ISO_TD R=%d > FC=%d\n", relative_frame_number, 

               frame_count);

        OHCI_SET_BM(iso_td.flags, TD_CC, OHCI_CC_DATAOVERRUN);

        ed->head &= ~OHCI_DPTR_MASK;

        ed->head |= (iso_td.next & OHCI_DPTR_MASK);

        iso_td.next = ohci->done;

        ohci->done = addr;

        i = OHCI_BM(iso_td.flags, TD_DI);

        if (i < ohci->done_count)

            ohci->done_count = i;

        ohci_put_iso_td(ohci, addr, &iso_td);

        return 0;

    }



    dir = OHCI_BM(ed->flags, ED_D);

    switch (dir) {

    case OHCI_TD_DIR_IN:

#ifdef DEBUG_ISOCH

        str = "in";

#endif

        pid = USB_TOKEN_IN;

        break;

    case OHCI_TD_DIR_OUT:

#ifdef DEBUG_ISOCH

        str = "out";

#endif

        pid = USB_TOKEN_OUT;

        break;

    case OHCI_TD_DIR_SETUP:

#ifdef DEBUG_ISOCH

        str = "setup";

#endif

        pid = USB_TOKEN_SETUP;

        break;

    default:

        printf("usb-ohci: Bad direction %d\n", dir);

        return 1;

    }



    if (!iso_td.bp || !iso_td.be) {

        printf("usb-ohci: ISO_TD bp 0x%.8x be 0x%.8x\n", iso_td.bp, iso_td.be);

        return 1;

    }



    start_offset = iso_td.offset[relative_frame_number];

    next_offset = iso_td.offset[relative_frame_number + 1];



    if (!(OHCI_BM(start_offset, TD_PSW_CC) & 0xe) || 

        ((relative_frame_number < frame_count) && 

         !(OHCI_BM(next_offset, TD_PSW_CC) & 0xe))) {

        printf("usb-ohci: ISO_TD cc != not accessed 0x%.8x 0x%.8x\n",

               start_offset, next_offset);

        return 1;

    }



    if ((relative_frame_number < frame_count) && (start_offset > next_offset)) {

        printf("usb-ohci: ISO_TD start_offset=0x%.8x > next_offset=0x%.8x\n",

                start_offset, next_offset);

        return 1;

    }



    if ((start_offset & 0x1000) == 0) {

        start_addr = (iso_td.bp & OHCI_PAGE_MASK) |

            (start_offset & OHCI_OFFSET_MASK);

    } else {

        start_addr = (iso_td.be & OHCI_PAGE_MASK) |

            (start_offset & OHCI_OFFSET_MASK);

    }



    if (relative_frame_number < frame_count) {

        end_offset = next_offset - 1;

        if ((end_offset & 0x1000) == 0) {

            end_addr = (iso_td.bp & OHCI_PAGE_MASK) |

                (end_offset & OHCI_OFFSET_MASK);

        } else {

            end_addr = (iso_td.be & OHCI_PAGE_MASK) |

                (end_offset & OHCI_OFFSET_MASK);

        }

    } else {

        /* Last packet in the ISO TD */

        end_addr = iso_td.be;

    }



    if ((start_addr & OHCI_PAGE_MASK) != (end_addr & OHCI_PAGE_MASK)) {

        len = (end_addr & OHCI_OFFSET_MASK) + 0x1001

            - (start_addr & OHCI_OFFSET_MASK);

    } else {

        len = end_addr - start_addr + 1;

    }



    if (len && dir != OHCI_TD_DIR_IN) {

        ohci_copy_iso_td(ohci, start_addr, end_addr, ohci->usb_buf, len, 0);

    }



    if (completion) {

        ret = ohci->usb_packet.len;

    } else {

        ret = USB_RET_NODEV;

        for (i = 0; i < ohci->num_ports; i++) {

            dev = ohci->rhport[i].port.dev;

            if ((ohci->rhport[i].ctrl & OHCI_PORT_PES) == 0)

                continue;

            ohci->usb_packet.pid = pid;

            ohci->usb_packet.devaddr = OHCI_BM(ed->flags, ED_FA);

            ohci->usb_packet.devep = OHCI_BM(ed->flags, ED_EN);

            ohci->usb_packet.data = ohci->usb_buf;

            ohci->usb_packet.len = len;

            ret = usb_handle_packet(dev, &ohci->usb_packet);

            if (ret != USB_RET_NODEV)

                break;

        }

    

        if (ret == USB_RET_ASYNC) {

            return 1;

        }

    }



#ifdef DEBUG_ISOCH

    printf("so 0x%.8x eo 0x%.8x\nsa 0x%.8x ea 0x%.8x\ndir %s len %zu ret %d\n",

           start_offset, end_offset, start_addr, end_addr, str, len, ret);

#endif



    /* Writeback */

    if (dir == OHCI_TD_DIR_IN && ret >= 0 && ret <= len) {

        /* IN transfer succeeded */

        ohci_copy_iso_td(ohci, start_addr, end_addr, ohci->usb_buf, ret, 1);

        OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_CC,

                    OHCI_CC_NOERROR);

        OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_SIZE, ret);

    } else if (dir == OHCI_TD_DIR_OUT && ret == len) {

        /* OUT transfer succeeded */

        OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_CC,

                    OHCI_CC_NOERROR);

        OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_SIZE, 0);

    } else {

        if (ret > (ssize_t) len) {

            printf("usb-ohci: DataOverrun %d > %zu\n", ret, len);

            OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_CC,

                        OHCI_CC_DATAOVERRUN);

            OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_SIZE,

                        len);

        } else if (ret >= 0) {

            printf("usb-ohci: DataUnderrun %d\n", ret);

            OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_CC,

                        OHCI_CC_DATAUNDERRUN);

        } else {

            switch (ret) {

            case USB_RET_NODEV:

                OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_CC,

                            OHCI_CC_DEVICENOTRESPONDING);

                OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_SIZE,

                            0);

                break;

            case USB_RET_NAK:

            case USB_RET_STALL:

                printf("usb-ohci: got NAK/STALL %d\n", ret);

                OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_CC,

                            OHCI_CC_STALL);

                OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_SIZE,

                            0);

                break;

            default:

                printf("usb-ohci: Bad device response %d\n", ret);

                OHCI_SET_BM(iso_td.offset[relative_frame_number], TD_PSW_CC,

                            OHCI_CC_UNDEXPETEDPID);

                break;

            }

        }

    }



    if (relative_frame_number == frame_count) {

        /* Last data packet of ISO TD - retire the TD to the Done Queue */

        OHCI_SET_BM(iso_td.flags, TD_CC, OHCI_CC_NOERROR);

        ed->head &= ~OHCI_DPTR_MASK;

        ed->head |= (iso_td.next & OHCI_DPTR_MASK);

        iso_td.next = ohci->done;

        ohci->done = addr;

        i = OHCI_BM(iso_td.flags, TD_DI);

        if (i < ohci->done_count)

            ohci->done_count = i;

    }

    ohci_put_iso_td(ohci, addr, &iso_td);

    return 1;

}
