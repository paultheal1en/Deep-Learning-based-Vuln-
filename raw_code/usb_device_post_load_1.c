static int usb_device_post_load(void *opaque, int version_id)

{

    USBDevice *dev = opaque;



    if (dev->state == USB_STATE_NOTATTACHED) {

        dev->attached = 0;

    } else {

        dev->attached = 1;






    return 0;
