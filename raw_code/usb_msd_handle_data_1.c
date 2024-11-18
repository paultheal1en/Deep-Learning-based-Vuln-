static int usb_msd_handle_data(USBDevice *dev, USBPacket *p)

{

    MSDState *s = (MSDState *)dev;

    int ret = 0;

    struct usb_msd_cbw cbw;

    uint8_t devep = p->devep;

    uint8_t *data = p->data;

    int len = p->len;



    switch (p->pid) {

    case USB_TOKEN_OUT:

        if (devep != 2)

            goto fail;



        switch (s->mode) {

        case USB_MSDM_CBW:

            if (len != 31) {

                fprintf(stderr, "usb-msd: Bad CBW size");

                goto fail;

            }

            memcpy(&cbw, data, 31);

            if (le32_to_cpu(cbw.sig) != 0x43425355) {

                fprintf(stderr, "usb-msd: Bad signature %08x\n",

                        le32_to_cpu(cbw.sig));

                goto fail;

            }

            DPRINTF("Command on LUN %d\n", cbw.lun);

            if (cbw.lun != 0) {

                fprintf(stderr, "usb-msd: Bad LUN %d\n", cbw.lun);

                goto fail;

            }

            s->tag = le32_to_cpu(cbw.tag);

            s->data_len = le32_to_cpu(cbw.data_len);

            if (s->data_len == 0) {

                s->mode = USB_MSDM_CSW;

            } else if (cbw.flags & 0x80) {

                s->mode = USB_MSDM_DATAIN;

            } else {

                s->mode = USB_MSDM_DATAOUT;

            }

            DPRINTF("Command tag 0x%x flags %08x len %d data %d\n",

                    s->tag, cbw.flags, cbw.cmd_len, s->data_len);

            s->residue = 0;

            s->scsi_len = 0;

            s->req = scsi_req_new(s->scsi_dev, s->tag, 0, NULL);

            scsi_req_enqueue(s->req, cbw.cmd);

            /* ??? Should check that USB and SCSI data transfer

               directions match.  */

            if (s->mode != USB_MSDM_CSW && s->residue == 0) {

                scsi_req_continue(s->req);

            }

            ret = len;

            break;



        case USB_MSDM_DATAOUT:

            DPRINTF("Data out %d/%d\n", len, s->data_len);

            if (len > s->data_len)

                goto fail;



            s->usb_buf = data;

            s->usb_len = len;

            if (s->scsi_len) {

                usb_msd_copy_data(s);

            }

            if (s->residue && s->usb_len) {

                s->data_len -= s->usb_len;

                if (s->data_len == 0)

                    s->mode = USB_MSDM_CSW;

                s->usb_len = 0;

            }

            if (s->usb_len) {

                DPRINTF("Deferring packet %p\n", p);

                s->packet = p;

                ret = USB_RET_ASYNC;

            } else {

                ret = len;

            }

            break;



        default:

            DPRINTF("Unexpected write (len %d)\n", len);

            goto fail;

        }

        break;



    case USB_TOKEN_IN:

        if (devep != 1)

            goto fail;



        switch (s->mode) {

        case USB_MSDM_DATAOUT:

            if (s->data_len != 0 || len < 13)

                goto fail;

            /* Waiting for SCSI write to complete.  */

            s->packet = p;

            ret = USB_RET_ASYNC;

            break;



        case USB_MSDM_CSW:

            DPRINTF("Command status %d tag 0x%x, len %d\n",

                    s->result, s->tag, len);

            if (len < 13)

                goto fail;



            usb_msd_send_status(s, p);

            s->mode = USB_MSDM_CBW;

            ret = 13;

            break;



        case USB_MSDM_DATAIN:

            DPRINTF("Data in %d/%d, scsi_len %d\n", len, s->data_len, s->scsi_len);

            if (len > s->data_len)

                len = s->data_len;

            s->usb_buf = data;

            s->usb_len = len;

            if (s->scsi_len) {

                usb_msd_copy_data(s);

            }

            if (s->residue && s->usb_len) {

                s->data_len -= s->usb_len;

                memset(s->usb_buf, 0, s->usb_len);

                if (s->data_len == 0)

                    s->mode = USB_MSDM_CSW;

                s->usb_len = 0;

            }

            if (s->usb_len) {

                DPRINTF("Deferring packet %p\n", p);

                s->packet = p;

                ret = USB_RET_ASYNC;

            } else {

                ret = len;

            }

            break;



        default:

            DPRINTF("Unexpected read (len %d)\n", len);

            goto fail;

        }

        break;



    default:

        DPRINTF("Bad token\n");

    fail:

        ret = USB_RET_STALL;

        break;

    }



    return ret;

}
