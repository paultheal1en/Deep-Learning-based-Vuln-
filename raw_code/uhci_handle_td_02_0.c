static int uhci_handle_td(UHCIState *s, UHCI_TD *td, uint32_t *int_mask,

                          int completion)

{

    uint8_t pid;

    int len = 0, max_len, err, ret = 0;



    /* ??? This is wrong for async completion.  */

    if (td->ctrl & TD_CTRL_IOC) {

        *int_mask |= 0x01;

    }



    if (!(td->ctrl & TD_CTRL_ACTIVE))

        return 1;



    /* TD is active */

    max_len = ((td->token >> 21) + 1) & 0x7ff;

    pid = td->token & 0xff;



    if (completion && (s->async_qh || s->async_frame_addr)) {

        ret = s->usb_packet.len;

        if (ret >= 0) {

            len = ret;

            if (len > max_len) {

                len = max_len;

                ret = USB_RET_BABBLE;

            }

            if (len > 0) {

                /* write the data back */

                cpu_physical_memory_write(td->buffer, s->usb_buf, len);

            }

        } else {

            len = 0;

        }

        s->async_qh = 0;

        s->async_frame_addr = 0;

    } else if (!completion) {

        s->usb_packet.pid = pid;

        s->usb_packet.devaddr = (td->token >> 8) & 0x7f;

        s->usb_packet.devep = (td->token >> 15) & 0xf;

        s->usb_packet.data = s->usb_buf;

        s->usb_packet.len = max_len;

        s->usb_packet.complete_cb = uhci_async_complete_packet;

        s->usb_packet.complete_opaque = s;

        switch(pid) {

        case USB_TOKEN_OUT:

        case USB_TOKEN_SETUP:

            cpu_physical_memory_read(td->buffer, s->usb_buf, max_len);

            ret = uhci_broadcast_packet(s, &s->usb_packet);

            len = max_len;

            break;

        case USB_TOKEN_IN:

            ret = uhci_broadcast_packet(s, &s->usb_packet);

            if (ret >= 0) {

                len = ret;

                if (len > max_len) {

                    len = max_len;

                    ret = USB_RET_BABBLE;

                }

                if (len > 0) {

                    /* write the data back */

                    cpu_physical_memory_write(td->buffer, s->usb_buf, len);

                }

            } else {

                len = 0;

            }

            break;

        default:

            /* invalid pid : frame interrupted */

            s->status |= UHCI_STS_HCPERR;

            uhci_update_irq(s);

            return -1;

        }

    }



    if (ret == USB_RET_ASYNC) {

        return 2;

    }

    if (td->ctrl & TD_CTRL_IOS)

        td->ctrl &= ~TD_CTRL_ACTIVE;

    if (ret >= 0) {

        td->ctrl = (td->ctrl & ~0x7ff) | ((len - 1) & 0x7ff);

        /* The NAK bit may have been set by a previous frame, so clear it

           here.  The docs are somewhat unclear, but win2k relies on this

           behavior.  */

        td->ctrl &= ~(TD_CTRL_ACTIVE | TD_CTRL_NAK);

        if (pid == USB_TOKEN_IN &&

            (td->ctrl & TD_CTRL_SPD) &&

            len < max_len) {

            *int_mask |= 0x02;

            /* short packet: do not update QH */

            return 1;

        } else {

            /* success */

            return 0;

        }

    } else {

        switch(ret) {

        default:

        case USB_RET_NODEV:

        do_timeout:

            td->ctrl |= TD_CTRL_TIMEOUT;

            err = (td->ctrl >> TD_CTRL_ERROR_SHIFT) & 3;

            if (err != 0) {

                err--;

                if (err == 0) {

                    td->ctrl &= ~TD_CTRL_ACTIVE;

                    s->status |= UHCI_STS_USBERR;

                    uhci_update_irq(s);

                }

            }

            td->ctrl = (td->ctrl & ~(3 << TD_CTRL_ERROR_SHIFT)) |

                (err << TD_CTRL_ERROR_SHIFT);

            return 1;

        case USB_RET_NAK:

            td->ctrl |= TD_CTRL_NAK;

            if (pid == USB_TOKEN_SETUP)

                goto do_timeout;

            return 1;

        case USB_RET_STALL:

            td->ctrl |= TD_CTRL_STALL;

            td->ctrl &= ~TD_CTRL_ACTIVE;

            return 1;

        case USB_RET_BABBLE:

            td->ctrl |= TD_CTRL_BABBLE | TD_CTRL_STALL;

            td->ctrl &= ~TD_CTRL_ACTIVE;

            /* frame interrupted */

            return -1;

        }

    }

}
