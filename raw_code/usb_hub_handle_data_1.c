static int usb_hub_handle_data(USBDevice *dev, USBPacket *p)

{

    USBHubState *s = (USBHubState *)dev;

    int ret;



    switch(p->pid) {

    case USB_TOKEN_IN:

        if (p->devep == 1) {

            USBHubPort *port;

            unsigned int status;

            int i, n;

            n = (NUM_PORTS + 1 + 7) / 8;

            if (p->len == 1) { /* FreeBSD workaround */

                n = 1;

            } else if (n > p->len) {

                return USB_RET_BABBLE;

            }

            status = 0;

            for(i = 0; i < NUM_PORTS; i++) {

                port = &s->ports[i];

                if (port->wPortChange)

                    status |= (1 << (i + 1));

            }

            if (status != 0) {

                for(i = 0; i < n; i++) {

                    p->data[i] = status >> (8 * i);

                }

                ret = n;

            } else {

                ret = USB_RET_NAK; /* usb11 11.13.1 */

            }

        } else {

            goto fail;

        }

        break;

    case USB_TOKEN_OUT:

    default:

    fail:

        ret = USB_RET_STALL;

        break;

    }

    return ret;

}
