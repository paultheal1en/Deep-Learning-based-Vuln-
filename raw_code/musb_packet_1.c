static void musb_packet(MUSBState *s, MUSBEndPoint *ep,

                int epnum, int pid, int len, USBCallback cb, int dir)

{

    int ret;

    int idx = epnum && dir;

    int ttype;



    /* ep->type[0,1] contains:

     * in bits 7:6 the speed (0 - invalid, 1 - high, 2 - full, 3 - slow)

     * in bits 5:4 the transfer type (BULK / INT)

     * in bits 3:0 the EP num

     */

    ttype = epnum ? (ep->type[idx] >> 4) & 3 : 0;



    ep->timeout[dir] = musb_timeout(ttype,

                    ep->type[idx] >> 6, ep->interval[idx]);

    ep->interrupt[dir] = ttype == USB_ENDPOINT_XFER_INT;

    ep->delayed_cb[dir] = cb;



    ep->packey[dir].p.pid = pid;

    /* A wild guess on the FADDR semantics... */

    ep->packey[dir].p.devaddr = ep->faddr[idx];

    ep->packey[dir].p.devep = ep->type[idx] & 0xf;

    ep->packey[dir].p.data = (void *) ep->buf[idx];

    ep->packey[dir].p.len = len;

    ep->packey[dir].ep = ep;

    ep->packey[dir].dir = dir;



    if (s->port.dev)

        ret = usb_handle_packet(s->port.dev, &ep->packey[dir].p);

    else

        ret = USB_RET_NODEV;



    if (ret == USB_RET_ASYNC) {

        ep->status[dir] = len;

        return;

    }



    ep->status[dir] = ret;

    musb_schedule_cb(&s->port, &ep->packey[dir].p);

}
