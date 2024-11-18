static int usb_hid_handle_control(USBDevice *dev, USBPacket *p,

               int request, int value, int index, int length, uint8_t *data)

{

    USBHIDState *us = DO_UPCAST(USBHIDState, dev, dev);

    HIDState *hs = &us->hid;

    int ret;



    ret = usb_desc_handle_control(dev, p, request, value, index, length, data);

    if (ret >= 0) {

        return ret;

    }



    ret = 0;

    switch (request) {

    case DeviceRequest | USB_REQ_GET_INTERFACE:

        data[0] = 0;

        ret = 1;

        break;

    case DeviceOutRequest | USB_REQ_SET_INTERFACE:

        ret = 0;

        break;

        /* hid specific requests */

    case InterfaceRequest | USB_REQ_GET_DESCRIPTOR:

        switch (value >> 8) {

        case 0x22:

            if (hs->kind == HID_MOUSE) {

		memcpy(data, qemu_mouse_hid_report_descriptor,

		       sizeof(qemu_mouse_hid_report_descriptor));

		ret = sizeof(qemu_mouse_hid_report_descriptor);

            } else if (hs->kind == HID_TABLET) {

                memcpy(data, qemu_tablet_hid_report_descriptor,

		       sizeof(qemu_tablet_hid_report_descriptor));

		ret = sizeof(qemu_tablet_hid_report_descriptor);

            } else if (hs->kind == HID_KEYBOARD) {

                memcpy(data, qemu_keyboard_hid_report_descriptor,

                       sizeof(qemu_keyboard_hid_report_descriptor));

                ret = sizeof(qemu_keyboard_hid_report_descriptor);

            }

            break;

        default:

            goto fail;

        }

        break;

    case GET_REPORT:

        if (hs->kind == HID_MOUSE || hs->kind == HID_TABLET) {

            ret = hid_pointer_poll(hs, data, length);

        } else if (hs->kind == HID_KEYBOARD) {

            ret = hid_keyboard_poll(hs, data, length);

        }

        us->changed = hs->n > 0;

        break;

    case SET_REPORT:

        if (hs->kind == HID_KEYBOARD) {

            ret = hid_keyboard_write(hs, data, length);

        } else {

            goto fail;

        }

        break;

    case GET_PROTOCOL:

        if (hs->kind != HID_KEYBOARD && hs->kind != HID_MOUSE) {

            goto fail;

        }

        ret = 1;

        data[0] = us->protocol;

        break;

    case SET_PROTOCOL:

        if (hs->kind != HID_KEYBOARD && hs->kind != HID_MOUSE) {

            goto fail;

        }

        ret = 0;

        us->protocol = value;

        break;

    case GET_IDLE:

        ret = 1;

        data[0] = us->idle;

        break;

    case SET_IDLE:

        us->idle = (uint8_t) (value >> 8);

        usb_hid_set_next_idle(us, qemu_get_clock_ns(vm_clock));

        ret = 0;

        break;

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
