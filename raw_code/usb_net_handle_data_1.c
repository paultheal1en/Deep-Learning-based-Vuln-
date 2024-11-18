static int usb_net_handle_data(USBDevice *dev, USBPacket *p)

{

    USBNetState *s = (USBNetState *) dev;

    int ret = 0;



    switch(p->pid) {

    case USB_TOKEN_IN:

        switch (p->devep) {

        case 1:

            ret = usb_net_handle_statusin(s, p);

            break;



        case 2:

            ret = usb_net_handle_datain(s, p);

            break;



        default:

            goto fail;

        }

        break;



    case USB_TOKEN_OUT:

        switch (p->devep) {

        case 2:

            ret = usb_net_handle_dataout(s, p);

            break;



        default:

            goto fail;

        }

        break;



    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    if (ret == USB_RET_STALL)

        fprintf(stderr, "usbnet: failed data transaction: "

                        "pid 0x%x ep 0x%x len 0x%x\n",

                        p->pid, p->devep, p->len);

    return ret;

}
