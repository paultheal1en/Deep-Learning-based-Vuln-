static int usb_hid_handle_data(USBDevice *dev, USBPacket *p)

{

    USBHIDState *s = (USBHIDState *)dev;

    int ret = 0;



    switch(p->pid) {

    case USB_TOKEN_IN:

        if (p->devep == 1) {

            int64_t curtime = qemu_get_clock_ns(vm_clock);

            if (!s->changed && (!s->idle || s->next_idle_clock - curtime > 0))

                return USB_RET_NAK;

            usb_hid_set_next_idle(s, curtime);

            if (s->kind == USB_MOUSE || s->kind == USB_TABLET) {

                ret = usb_pointer_poll(s, p->data, p->len);

            }

            else if (s->kind == USB_KEYBOARD) {

                ret = usb_keyboard_poll(s, p->data, p->len);

            }

            s->changed = s->n > 0;

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
