static void xhci_check_iso_kick(XHCIState *xhci, XHCITransfer *xfer,

                                XHCIEPContext *epctx, uint64_t mfindex)

{

    if (xfer->mfindex_kick > mfindex) {

        timer_mod(epctx->kick_timer, qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +

                       (xfer->mfindex_kick - mfindex) * 125000);

        xfer->running_retry = 1;

    } else {

        epctx->mfindex_last = xfer->mfindex_kick;

        timer_del(epctx->kick_timer);

        xfer->running_retry = 0;

    }

}
