static int usb_serial_handle_control(USBDevice *dev, int request, int value,

                                  int index, int length, uint8_t *data)

{

    USBSerialState *s = (USBSerialState *)dev;

    int ret;



    DPRINTF("got control %x, value %x\n",request, value);

    ret = usb_desc_handle_control(dev, request, value, index, length, data);

    if (ret >= 0) {

        return ret;

    }



    ret = 0;

    switch (request) {

    case DeviceRequest | USB_REQ_GET_STATUS:

        data[0] = (0 << USB_DEVICE_SELF_POWERED) |

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

    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        ret = 0;

        break;

    case EndpointOutRequest | USB_REQ_CLEAR_FEATURE:

        ret = 0;

        break;



        /* Class specific requests.  */

    case DeviceOutVendor | FTDI_RESET:

        switch (value) {

        case FTDI_RESET_SIO:

            usb_serial_reset(s);

            break;

        case FTDI_RESET_RX:

            s->recv_ptr = 0;

            s->recv_used = 0;

            /* TODO: purge from char device */

            break;

        case FTDI_RESET_TX:

            /* TODO: purge from char device */

            break;

        }

        break;

    case DeviceOutVendor | FTDI_SET_MDM_CTRL:

    {

        static int flags;

        qemu_chr_ioctl(s->cs,CHR_IOCTL_SERIAL_GET_TIOCM, &flags);

        if (value & FTDI_SET_RTS) {

            if (value & FTDI_RTS)

                flags |= CHR_TIOCM_RTS;

            else

                flags &= ~CHR_TIOCM_RTS;

        }

        if (value & FTDI_SET_DTR) {

            if (value & FTDI_DTR)

                flags |= CHR_TIOCM_DTR;

            else

                flags &= ~CHR_TIOCM_DTR;

        }

        qemu_chr_ioctl(s->cs,CHR_IOCTL_SERIAL_SET_TIOCM, &flags);

        break;

    }

    case DeviceOutVendor | FTDI_SET_FLOW_CTRL:

        /* TODO: ioctl */

        break;

    case DeviceOutVendor | FTDI_SET_BAUD: {

        static const int subdivisors8[8] = { 0, 4, 2, 1, 3, 5, 6, 7 };

        int subdivisor8 = subdivisors8[((value & 0xc000) >> 14)

                                     | ((index & 1) << 2)];

        int divisor = value & 0x3fff;



        /* chip special cases */

        if (divisor == 1 && subdivisor8 == 0)

            subdivisor8 = 4;

        if (divisor == 0 && subdivisor8 == 0)

            divisor = 1;



        s->params.speed = (48000000 / 2) / (8 * divisor + subdivisor8);

        qemu_chr_ioctl(s->cs, CHR_IOCTL_SERIAL_SET_PARAMS, &s->params);

        break;

    }

    case DeviceOutVendor | FTDI_SET_DATA:

        switch (value & FTDI_PARITY) {

            case 0:

                s->params.parity = 'N';

                break;

            case FTDI_ODD:

                s->params.parity = 'O';

                break;

            case FTDI_EVEN:

                s->params.parity = 'E';

                break;

            default:

                DPRINTF("unsupported parity %d\n", value & FTDI_PARITY);

                goto fail;

        }

        switch (value & FTDI_STOP) {

            case FTDI_STOP1:

                s->params.stop_bits = 1;

                break;

            case FTDI_STOP2:

                s->params.stop_bits = 2;

                break;

            default:

                DPRINTF("unsupported stop bits %d\n", value & FTDI_STOP);

                goto fail;

        }

        qemu_chr_ioctl(s->cs, CHR_IOCTL_SERIAL_SET_PARAMS, &s->params);

        /* TODO: TX ON/OFF */

        break;

    case DeviceInVendor | FTDI_GET_MDM_ST:

        data[0] = usb_get_modem_lines(s) | 1;

        data[1] = 0;

        ret = 2;

        break;

    case DeviceOutVendor | FTDI_SET_EVENT_CHR:

        /* TODO: handle it */

        s->event_chr = value;

        break;

    case DeviceOutVendor | FTDI_SET_ERROR_CHR:

        /* TODO: handle it */

        s->error_chr = value;

        break;

    case DeviceOutVendor | FTDI_SET_LATENCY:

        s->latency = value;

        break;

    case DeviceInVendor | FTDI_GET_LATENCY:

        data[0] = s->latency;

        ret = 1;

        break;

    default:

    fail:

        DPRINTF("got unsupported/bogus control %x, value %x\n", request, value);

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
