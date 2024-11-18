int usb_desc_handle_control(USBDevice *dev, int request, int value,

                            int index, int length, uint8_t *data)

{

    const USBDesc *desc = dev->info->usb_desc;

    int ret = -1;



    assert(desc != NULL);

    switch(request) {

    case DeviceOutRequest | USB_REQ_SET_ADDRESS:

        dev->addr = value;

        trace_usb_set_addr(dev->addr);

        ret = 0;

        break;



    case DeviceRequest | USB_REQ_GET_DESCRIPTOR:

        ret = usb_desc_get_descriptor(dev, value, data, length);

        break;

    }

    return ret;

}
