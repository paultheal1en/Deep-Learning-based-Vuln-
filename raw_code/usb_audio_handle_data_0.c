static int usb_audio_handle_data(USBDevice *dev, USBPacket *p)

{

    USBAudioState *s = (USBAudioState *) dev;

    int ret = 0;



    switch (p->pid) {

    case USB_TOKEN_OUT:

        switch (p->devep) {

        case 1:

            ret = usb_audio_handle_dataout(s, p);

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

    if (ret == USB_RET_STALL && s->debug) {

        fprintf(stderr, "usb-audio: failed data transaction: "

                        "pid 0x%x ep 0x%x len 0x%zx\n",

                        p->pid, p->devep, p->iov.size);

    }

    return ret;

}
