static TRBCCode xhci_reset_ep(XHCIState *xhci, unsigned int slotid,

                              unsigned int epid)

{

    XHCISlot *slot;

    XHCIEPContext *epctx;

    USBDevice *dev;



    trace_usb_xhci_ep_reset(slotid, epid);

    assert(slotid >= 1 && slotid <= xhci->numslots);



    if (epid < 1 || epid > 31) {

        fprintf(stderr, "xhci: bad ep %d\n", epid);

        return CC_TRB_ERROR;

    }



    slot = &xhci->slots[slotid-1];



    if (!slot->eps[epid-1]) {

        DPRINTF("xhci: slot %d ep %d not enabled\n", slotid, epid);

        return CC_EP_NOT_ENABLED_ERROR;

    }



    epctx = slot->eps[epid-1];



    if (epctx->state != EP_HALTED) {

        fprintf(stderr, "xhci: reset EP while EP %d not halted (%d)\n",

                epid, epctx->state);

        return CC_CONTEXT_STATE_ERROR;

    }



    if (xhci_ep_nuke_xfers(xhci, slotid, epid) > 0) {

        fprintf(stderr, "xhci: FIXME: endpoint reset w/ xfers running, "

                "data might be lost\n");

    }



    uint8_t ep = epid>>1;



    if (epid & 1) {

        ep |= 0x80;

    }



    dev = xhci->slots[slotid-1].uport->dev;

    if (!dev) {

        return CC_USB_TRANSACTION_ERROR;

    }



    xhci_set_ep_state(xhci, epctx, NULL, EP_STOPPED);



    if (epctx->nr_pstreams) {

        xhci_reset_streams(epctx);

    }



    return CC_SUCCESS;

}
