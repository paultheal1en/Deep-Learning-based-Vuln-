static void usb_host_handle_control(USBDevice *udev, USBPacket *p,

                                    int request, int value, int index,

                                    int length, uint8_t *data)

{

    USBHostDevice *s = USB_HOST_DEVICE(udev);

    USBHostRequest *r;

    int rc;



    trace_usb_host_req_control(s->bus_num, s->addr, p, request, value, index);



    if (s->dh == NULL) {

        p->status = USB_RET_NODEV;

        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);

        return;

    }



    switch (request) {

    case DeviceOutRequest | USB_REQ_SET_ADDRESS:

        usb_host_set_address(s, value);

        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);

        return;



    case DeviceOutRequest | USB_REQ_SET_CONFIGURATION:

        usb_host_set_config(s, value & 0xff, p);

        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);

        return;



    case InterfaceOutRequest | USB_REQ_SET_INTERFACE:

        usb_host_set_interface(s, index, value, p);

        trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);

        return;



    case EndpointOutRequest | USB_REQ_CLEAR_FEATURE:

        if (value == 0) { /* clear halt */

            int pid = (index & USB_DIR_IN) ? USB_TOKEN_IN : USB_TOKEN_OUT;

            libusb_clear_halt(s->dh, index);

            usb_ep_set_halted(udev, pid, index & 0x0f, 0);

            trace_usb_host_req_emulated(s->bus_num, s->addr, p, p->status);

            return;

        }

    }



    r = usb_host_req_alloc(s, p, (request >> 8) & USB_DIR_IN, length + 8);

    r->cbuf = data;

    r->clen = length;

    memcpy(r->buffer, udev->setup_buf, 8);

    if (!r->in) {

        memcpy(r->buffer + 8, r->cbuf, r->clen);

    }



    /* Fix up USB-3 ep0 maxpacket size to allow superspeed connected devices

     * to work redirected to a not superspeed capable hcd */

    if (udev->speed == USB_SPEED_SUPER &&

        !(udev->port->speedmask & USB_SPEED_MASK_SUPER) &&

        request == 0x8006 && value == 0x100 && index == 0) {

        r->usb3ep0quirk = true;

    }



    libusb_fill_control_transfer(r->xfer, s->dh, r->buffer,

                                 usb_host_req_complete_ctrl, r,

                                 CONTROL_TIMEOUT);

    rc = libusb_submit_transfer(r->xfer);

    if (rc != 0) {

        p->status = USB_RET_NODEV;

        trace_usb_host_req_complete(s->bus_num, s->addr, p,

                                    p->status, p->actual_length);

        if (rc == LIBUSB_ERROR_NO_DEVICE) {

            usb_host_nodev(s);

        }

        return;

    }



    p->status = USB_RET_ASYNC;

}
