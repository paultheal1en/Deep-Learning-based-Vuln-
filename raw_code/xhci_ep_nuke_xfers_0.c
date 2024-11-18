static int xhci_ep_nuke_xfers(XHCIState *xhci, unsigned int slotid,

                               unsigned int epid, TRBCCode report)

{

    XHCISlot *slot;

    XHCIEPContext *epctx;

    int i, xferi, killed = 0;

    USBEndpoint *ep = NULL;

    assert(slotid >= 1 && slotid <= xhci->numslots);

    assert(epid >= 1 && epid <= 31);



    DPRINTF("xhci_ep_nuke_xfers(%d, %d)\n", slotid, epid);



    slot = &xhci->slots[slotid-1];



    if (!slot->eps[epid-1]) {

        return 0;

    }



    epctx = slot->eps[epid-1];



    xferi = epctx->next_xfer;

    for (i = 0; i < TD_QUEUE; i++) {

        killed += xhci_ep_nuke_one_xfer(&epctx->transfers[xferi], report);

        if (killed) {

            report = 0; /* Only report once */

        }

        epctx->transfers[xferi].packet.ep = NULL;

        xferi = (xferi + 1) % TD_QUEUE;

    }



    ep = xhci_epid_to_usbep(xhci, slotid, epid);

    if (ep) {

        usb_device_ep_stopped(ep->dev, ep);

    }

    return killed;

}
