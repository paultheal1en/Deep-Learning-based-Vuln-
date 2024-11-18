static int usb_bt_handle_data(USBDevice *dev, USBPacket *p)

{

    struct USBBtState *s = (struct USBBtState *) dev->opaque;

    int ret = 0;



    if (!s->config)

        goto fail;



    switch (p->pid) {

    case USB_TOKEN_IN:

        switch (p->devep & 0xf) {

        case USB_EVT_EP:

            ret = usb_bt_fifo_dequeue(&s->evt, p);

            break;



        case USB_ACL_EP:

            ret = usb_bt_fifo_dequeue(&s->acl, p);

            break;



        case USB_SCO_EP:

            ret = usb_bt_fifo_dequeue(&s->sco, p);

            break;



        default:

            goto fail;

        }

        break;



    case USB_TOKEN_OUT:

        switch (p->devep & 0xf) {

        case USB_ACL_EP:

            usb_bt_fifo_out_enqueue(s, &s->outacl, s->hci->acl_send,

                            usb_bt_hci_acl_complete, p->data, p->len);

            break;



        case USB_SCO_EP:

            usb_bt_fifo_out_enqueue(s, &s->outsco, s->hci->sco_send,

                            usb_bt_hci_sco_complete, p->data, p->len);

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



    return ret;

}
