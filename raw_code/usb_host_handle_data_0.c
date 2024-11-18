static int usb_host_handle_data(USBDevice *dev, USBPacket *p)

{

    USBHostDevice *s = DO_UPCAST(USBHostDevice, dev, dev);

    struct usbdevfs_urb *urb;

    AsyncURB *aurb;

    int ret, rem, prem, v;

    uint8_t *pbuf;

    uint8_t ep;



    trace_usb_host_req_data(s->bus_num, s->addr,

                            p->pid == USB_TOKEN_IN,

                            p->devep, p->iov.size);



    if (!is_valid(s, p->pid, p->devep)) {

        trace_usb_host_req_complete(s->bus_num, s->addr, USB_RET_NAK);

        return USB_RET_NAK;

    }



    if (p->pid == USB_TOKEN_IN) {

        ep = p->devep | 0x80;

    } else {

        ep = p->devep;

    }



    if (is_halted(s, p->pid, p->devep)) {

        unsigned int arg = ep;

        ret = ioctl(s->fd, USBDEVFS_CLEAR_HALT, &arg);

        if (ret < 0) {

            perror("USBDEVFS_CLEAR_HALT");

            trace_usb_host_req_complete(s->bus_num, s->addr, USB_RET_NAK);

            return USB_RET_NAK;

        }

        clear_halt(s, p->pid, p->devep);

    }



    if (is_isoc(s, p->pid, p->devep)) {

        return usb_host_handle_iso_data(s, p, p->pid == USB_TOKEN_IN);

    }



    v = 0;

    prem = p->iov.iov[v].iov_len;

    pbuf = p->iov.iov[v].iov_base;

    rem = p->iov.size;

    while (rem) {

        if (prem == 0) {

            v++;

            assert(v < p->iov.niov);

            prem = p->iov.iov[v].iov_len;

            pbuf = p->iov.iov[v].iov_base;

            assert(prem <= rem);

        }

        aurb = async_alloc(s);

        aurb->packet = p;



        urb = &aurb->urb;

        urb->endpoint      = ep;

        urb->type          = usb_host_usbfs_type(s, p);

        urb->usercontext   = s;

        urb->buffer        = pbuf;

        urb->buffer_length = prem;



        if (urb->buffer_length > MAX_USBFS_BUFFER_SIZE) {

            urb->buffer_length = MAX_USBFS_BUFFER_SIZE;

        }

        pbuf += urb->buffer_length;

        prem -= urb->buffer_length;

        rem  -= urb->buffer_length;

        if (rem) {

            aurb->more         = 1;

        }



        trace_usb_host_urb_submit(s->bus_num, s->addr, aurb,

                                  urb->buffer_length, aurb->more);

        ret = ioctl(s->fd, USBDEVFS_SUBMITURB, urb);



        DPRINTF("husb: data submit: ep 0x%x, len %u, more %d, packet %p, aurb %p\n",

                urb->endpoint, urb->buffer_length, aurb->more, p, aurb);



        if (ret < 0) {

            perror("USBDEVFS_SUBMITURB");

            async_free(aurb);



            switch(errno) {

            case ETIMEDOUT:

                trace_usb_host_req_complete(s->bus_num, s->addr, USB_RET_NAK);

                return USB_RET_NAK;

            case EPIPE:

            default:

                trace_usb_host_req_complete(s->bus_num, s->addr, USB_RET_STALL);

                return USB_RET_STALL;

            }

        }

    }



    return USB_RET_ASYNC;

}
