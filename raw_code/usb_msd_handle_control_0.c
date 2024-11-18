static int usb_msd_handle_control(USBDevice *dev, int request, int value,

                                  int index, int length, uint8_t *data)

{

    MSDState *s = (MSDState *)dev;

    int ret;



    ret = usb_desc_handle_control(dev, request, value, index, length, data);

    if (ret >= 0) {

        return ret;

    }



    ret = 0;

    switch (request) {

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

    case EndpointOutRequest | USB_REQ_CLEAR_FEATURE:

        ret = 0;

        break;

    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        ret = 0;

        break;

        /* Class specific requests.  */

    case ClassInterfaceOutRequest | MassStorageReset:

        /* Reset state ready for the next CBW.  */

        s->mode = USB_MSDM_CBW;

        ret = 0;

        break;

    case ClassInterfaceRequest | GetMaxLun:

        data[0] = 0;

        ret = 1;

        break;

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
