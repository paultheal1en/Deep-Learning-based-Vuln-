static int usb_host_usbfs_type(USBHostDevice *s, USBPacket *p)

{

    static const int usbfs[] = {

        [USB_ENDPOINT_XFER_CONTROL] = USBDEVFS_URB_TYPE_CONTROL,

        [USB_ENDPOINT_XFER_ISOC]    = USBDEVFS_URB_TYPE_ISO,

        [USB_ENDPOINT_XFER_BULK]    = USBDEVFS_URB_TYPE_BULK,

        [USB_ENDPOINT_XFER_INT]     = USBDEVFS_URB_TYPE_INTERRUPT,

    };

    uint8_t type = usb_ep_get_type(&s->dev, p->pid, p->devep);

    assert(type < ARRAY_SIZE(usbfs));

    return usbfs[type];

}
