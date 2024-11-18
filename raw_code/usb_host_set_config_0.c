static void usb_host_set_config(USBHostDevice *s, int config, USBPacket *p)

{

    int rc;



    trace_usb_host_set_config(s->bus_num, s->addr, config);



    usb_host_release_interfaces(s);

    usb_host_detach_kernel(s);

    rc = libusb_set_configuration(s->dh, config);

    if (rc != 0) {

        usb_host_libusb_error("libusb_set_configuration", rc);

        p->status = USB_RET_STALL;

        if (rc == LIBUSB_ERROR_NO_DEVICE) {

            usb_host_nodev(s);

        }

        return;

    }

    p->status = usb_host_claim_interfaces(s, config);

    if (p->status != USB_RET_SUCCESS) {

        return;

    }

    usb_host_ep_update(s);

}
