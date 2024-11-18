static void usb_msd_send_status(MSDState *s, USBPacket *p)

{

    struct usb_msd_csw csw;

    int len;



    csw.sig = cpu_to_le32(0x53425355);

    csw.tag = cpu_to_le32(s->tag);

    csw.residue = s->residue;

    csw.status = s->result;



    len = MIN(sizeof(csw), p->len);

    memcpy(p->data, &csw, len);

}
