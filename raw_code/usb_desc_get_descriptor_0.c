int usb_desc_get_descriptor(USBDevice *dev, int value, uint8_t *dest, size_t len)

{

    const USBDesc *desc = dev->info->usb_desc;

    uint8_t buf[256];

    uint8_t type = value >> 8;

    uint8_t index = value & 0xff;

    int ret = -1;



    switch(type) {

    case USB_DT_DEVICE:

        ret = usb_desc_device(&desc->id, desc->full, buf, sizeof(buf));

        trace_usb_desc_device(dev->addr, len, ret);

        break;

    case USB_DT_CONFIG:

        if (index < desc->full->bNumConfigurations) {

            ret = usb_desc_config(desc->full->confs + index, buf, sizeof(buf));

        }

        trace_usb_desc_config(dev->addr, index, len, ret);

        break;

    case USB_DT_STRING:

        ret = usb_desc_string(dev, index, buf, sizeof(buf));

        trace_usb_desc_string(dev->addr, index, len, ret);

        break;

    default:

        fprintf(stderr, "%s: %d unknown type %d (len %zd)\n", __FUNCTION__,

                dev->addr, type, len);

        break;

    }



    if (ret > 0) {

        if (ret > len) {

            ret = len;

        }

        memcpy(dest, buf, ret);

    }

    return ret;

}
