static int usb_hid_handle_data(USBDevice *dev, USBPacket *p)

{

    USBHIDState *us = DO_UPCAST(USBHIDState, dev, dev);

    HIDState *hs = &us->hid;

    uint8_t buf[p->iov.size];

    int ret = 0;



    switch (p->pid) {

    case USB_TOKEN_IN:

        if (p->devep == 1) {

            int64_t curtime = qemu_get_clock_ns(vm_clock);

            if (!us->changed &&

                (!us->idle || us->next_idle_clock - curtime > 0)) {

                return USB_RET_NAK;

            }

            usb_hid_set_next_idle(us, curtime);

            if (hs->kind == HID_MOUSE || hs->kind == HID_TABLET) {

                ret = hid_pointer_poll(hs, buf, p->iov.size);

            } else if (hs->kind == HID_KEYBOARD) {

                ret = hid_keyboard_poll(hs, buf, p->iov.size);

            }

            usb_packet_copy(p, buf, ret);

            us->changed = hs->n > 0;

        } else {

            goto fail;

        }

        break;

    case USB_TOKEN_OUT:

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
