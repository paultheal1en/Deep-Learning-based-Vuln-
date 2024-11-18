static int xhci_fire_ctl_transfer(XHCIState *xhci, XHCITransfer *xfer)

{

    XHCITRB *trb_setup, *trb_status;

    uint8_t bmRequestType;



    trb_setup = &xfer->trbs[0];

    trb_status = &xfer->trbs[xfer->trb_count-1];



    trace_usb_xhci_xfer_start(xfer, xfer->epctx->slotid,

                              xfer->epctx->epid, xfer->streamid);



    /* at most one Event Data TRB allowed after STATUS */

    if (TRB_TYPE(*trb_status) == TR_EVDATA && xfer->trb_count > 2) {

        trb_status--;

    }



    /* do some sanity checks */

    if (TRB_TYPE(*trb_setup) != TR_SETUP) {

        DPRINTF("xhci: ep0 first TD not SETUP: %d\n",

                TRB_TYPE(*trb_setup));

        return -1;

    }

    if (TRB_TYPE(*trb_status) != TR_STATUS) {

        DPRINTF("xhci: ep0 last TD not STATUS: %d\n",

                TRB_TYPE(*trb_status));

        return -1;

    }

    if (!(trb_setup->control & TRB_TR_IDT)) {

        DPRINTF("xhci: Setup TRB doesn't have IDT set\n");

        return -1;

    }

    if ((trb_setup->status & 0x1ffff) != 8) {

        DPRINTF("xhci: Setup TRB has bad length (%d)\n",

                (trb_setup->status & 0x1ffff));

        return -1;

    }



    bmRequestType = trb_setup->parameter;



    xfer->in_xfer = bmRequestType & USB_DIR_IN;

    xfer->iso_xfer = false;

    xfer->timed_xfer = false;



    if (xhci_setup_packet(xfer) < 0) {

        return -1;

    }

    xfer->packet.parameter = trb_setup->parameter;



    usb_handle_packet(xfer->packet.ep->dev, &xfer->packet);



    xhci_try_complete_packet(xfer);

    if (!xfer->running_async && !xfer->running_retry) {

        xhci_kick_epctx(xfer->epctx, 0);

    }

    return 0;

}
