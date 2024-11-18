static int usb_net_handle_datain(USBNetState *s, USBPacket *p)

{

    int ret = USB_RET_NAK;



    if (s->in_ptr > s->in_len) {

        s->in_ptr = s->in_len = 0;

        ret = USB_RET_NAK;

        return ret;

    }

    if (!s->in_len) {

        ret = USB_RET_NAK;

        return ret;

    }

    ret = s->in_len - s->in_ptr;

    if (ret > p->iov.size) {

        ret = p->iov.size;

    }

    usb_packet_copy(p, &s->in_buf[s->in_ptr], ret);

    s->in_ptr += ret;

    if (s->in_ptr >= s->in_len &&

                    (is_rndis(s) || (s->in_len & (64 - 1)) || !ret)) {

        /* no short packet necessary */

        s->in_ptr = s->in_len = 0;

    }



#ifdef TRAFFIC_DEBUG

    fprintf(stderr, "usbnet: data in len %zu return %d", p->iov.size, ret);

    iov_hexdump(p->iov.iov, p->iov.niov, stderr, "usbnet", ret);

#endif



    return ret;

}
