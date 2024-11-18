static int usbredir_post_load(void *priv, int version_id)
{
    USBRedirDevice *dev = priv;
    switch (dev->device_info.speed) {
    case usb_redir_speed_low:
        dev->dev.speed = USB_SPEED_LOW;
        break;
    case usb_redir_speed_full:
        dev->dev.speed = USB_SPEED_FULL;
        break;
    case usb_redir_speed_high:
        dev->dev.speed = USB_SPEED_HIGH;
        break;
    case usb_redir_speed_super:
        dev->dev.speed = USB_SPEED_SUPER;
        break;
    default:
        dev->dev.speed = USB_SPEED_FULL;
    dev->dev.speedmask = (1 << dev->dev.speed);
    usbredir_setup_usb_eps(dev);
    usbredir_check_bulk_receiving(dev);