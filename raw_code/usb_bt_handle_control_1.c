static int usb_bt_handle_control(USBDevice *dev, USBPacket *p,

               int request, int value, int index, int length, uint8_t *data)

{

    struct USBBtState *s = (struct USBBtState *) dev->opaque;

    int ret;



    ret = usb_desc_handle_control(dev, p, request, value, index, length, data);

    if (ret >= 0) {

        switch (request) {

        case DeviceRequest | USB_REQ_GET_CONFIGURATION:

            s->config = 0;

            break;

        case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

            s->config = 1;

            usb_bt_fifo_reset(&s->evt);

            usb_bt_fifo_reset(&s->acl);

            usb_bt_fifo_reset(&s->sco);

            break;

        }

        return ret;

    }



    ret = 0;

    switch (request) {

    case InterfaceRequest | USB_REQ_GET_STATUS:

    case EndpointRequest | USB_REQ_GET_STATUS:

        data[0] = 0x00;

        data[1] = 0x00;

        ret = 2;

        break;

    case InterfaceOutRequest | USB_REQ_CLEAR_FEATURE:

    case EndpointOutRequest | USB_REQ_CLEAR_FEATURE:

        goto fail;

    case InterfaceOutRequest | USB_REQ_SET_FEATURE:

    case EndpointOutRequest | USB_REQ_SET_FEATURE:

        goto fail;

        break;

    case InterfaceRequest | USB_REQ_GET_INTERFACE:

        if (value != 0 || (index & ~1) || length != 1)

            goto fail;

        if (index == 1)

            data[0] = s->altsetting;

        else

            data[0] = 0;

        ret = 1;

        break;

    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        if ((index & ~1) || length != 0 ||

                        (index == 1 && (value < 0 || value > 4)) ||

                        (index == 0 && value != 0)) {

            printf("%s: Wrong SET_INTERFACE request (%i, %i)\n",

                            __FUNCTION__, index, value);

            goto fail;

        }

        s->altsetting = value;

        ret = 0;

        break;

    case ((USB_DIR_OUT | USB_TYPE_CLASS | USB_RECIP_DEVICE) << 8):

        if (s->config)

            usb_bt_fifo_out_enqueue(s, &s->outcmd, s->hci->cmd_send,

                            usb_bt_hci_cmd_complete, data, length);

        break;

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
