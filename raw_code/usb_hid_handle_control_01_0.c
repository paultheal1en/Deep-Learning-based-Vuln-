static int usb_hid_handle_control(USBDevice *dev, int request, int value,

                                  int index, int length, uint8_t *data)

{

    USBHIDState *s = (USBHIDState *)dev;

    int ret;



    ret = usb_desc_handle_control(dev, request, value, index, length, data);

    if (ret >= 0) {

        return ret;

    }



    ret = 0;

    switch(request) {

    case DeviceRequest | USB_REQ_GET_STATUS:

        data[0] = (1 << USB_DEVICE_SELF_POWERED) |

            (dev->remote_wakeup << USB_DEVICE_REMOTE_WAKEUP);

        data[1] = 0x00;

        ret = 2;

        break;

    case DeviceOutRequest | USB_REQ_CLEAR_FEATURE:

        if (value == USB_DEVICE_REMOTE_WAKEUP) {

            dev->remote_wakeup = 0;

        } else {

            goto fail;

        }

        ret = 0;

        break;

    case DeviceOutRequest | USB_REQ_SET_FEATURE:

        if (value == USB_DEVICE_REMOTE_WAKEUP) {

            dev->remote_wakeup = 1;

        } else {

            goto fail;

        }

        ret = 0;

        break;

    case DeviceRequest | USB_REQ_GET_CONFIGURATION:

        data[0] = 1;

        ret = 1;

        break;

    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

        ret = 0;

        break;

    case DeviceRequest | USB_REQ_GET_INTERFACE:

        data[0] = 0;

        ret = 1;

        break;

    case DeviceOutRequest | USB_REQ_SET_INTERFACE:

        ret = 0;

        break;

        /* hid specific requests */

    case InterfaceRequest | USB_REQ_GET_DESCRIPTOR:

        switch(value >> 8) {

        case 0x22:

	    if (s->kind == USB_MOUSE) {

		memcpy(data, qemu_mouse_hid_report_descriptor,

		       sizeof(qemu_mouse_hid_report_descriptor));

		ret = sizeof(qemu_mouse_hid_report_descriptor);

	    } else if (s->kind == USB_TABLET) {

		memcpy(data, qemu_tablet_hid_report_descriptor,

		       sizeof(qemu_tablet_hid_report_descriptor));

		ret = sizeof(qemu_tablet_hid_report_descriptor);

            } else if (s->kind == USB_KEYBOARD) {

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

	if (s->kind == USB_MOUSE)

            ret = usb_mouse_poll(s, data, length);

	else if (s->kind == USB_TABLET)

            ret = usb_tablet_poll(s, data, length);

        else if (s->kind == USB_KEYBOARD)

            ret = usb_keyboard_poll(&s->kbd, data, length);

        break;

    case SET_REPORT:

        if (s->kind == USB_KEYBOARD)

            ret = usb_keyboard_write(&s->kbd, data, length);

        else

            goto fail;

        break;

    case GET_PROTOCOL:

        if (s->kind != USB_KEYBOARD)

            goto fail;

        ret = 1;

        data[0] = s->protocol;

        break;

    case SET_PROTOCOL:

        if (s->kind != USB_KEYBOARD)

            goto fail;

        ret = 0;

        s->protocol = value;

        break;

    case GET_IDLE:

        ret = 1;

        data[0] = s->idle;

        break;

    case SET_IDLE:

        s->idle = (uint8_t) (value >> 8);

        usb_hid_set_next_idle(s, qemu_get_clock(vm_clock));

        ret = 0;

        break;

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
