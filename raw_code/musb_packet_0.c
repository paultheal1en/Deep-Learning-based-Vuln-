static void musb_packet(MUSBState *s, MUSBEndPoint *ep,

                int epnum, int pid, int len, USBCallback cb, int dir)

{

    USBDevice *dev;

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



    /* A wild guess on the FADDR semantics... */

    usb_packet_setup(&ep->packey[dir].p, pid, ep->faddr[idx],

                     ep->type[idx] & 0xf);

    usb_packet_addbuf(&ep->packey[dir].p, ep->buf[idx], len);

    ep->packey[dir].ep = ep;

    ep->packey[dir].dir = dir;



    dev = usb_find_device(&s->port, ep->packey[dir].p.devaddr);

    ret = usb_handle_packet(dev, &ep->packey[dir].p);



    if (ret == USB_RET_ASYNC) {

        ep->status[dir] = len;

        return;

    }



    ep->status[dir] = ret;

    musb_schedule_cb(&s->port, &ep->packey[dir].p);

}
