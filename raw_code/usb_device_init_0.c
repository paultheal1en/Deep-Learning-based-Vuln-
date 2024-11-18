static int usb_device_init(USBDevice *dev)

{

    USBDeviceClass *klass = USB_DEVICE_GET_CLASS(dev);

    if (klass->init) {

        return klass->init(dev);

    }

    return 0;

}
