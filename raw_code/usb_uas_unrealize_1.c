static void usb_uas_unrealize(USBDevice *dev, Error **errp)
{
    UASDevice *uas = USB_UAS(dev);
    qemu_bh_delete(uas->status_bh);
}