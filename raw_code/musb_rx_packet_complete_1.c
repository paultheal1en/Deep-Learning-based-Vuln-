static void musb_rx_packet_complete(USBPacket *packey, void *opaque)

{

    /* Unfortunately we can't use packey->devep because that's the remote

     * endpoint number and may be different than our local.  */

    MUSBEndPoint *ep = (MUSBEndPoint *) opaque;

    int epnum = ep->epnum;

    MUSBState *s = ep->musb;



    ep->fifostart[1] = 0;

    ep->fifolen[1] = 0;



#ifdef CLEAR_NAK

    if (ep->status[1] != USB_RET_NAK) {

#endif

        ep->csr[1] &= ~MGC_M_RXCSR_H_REQPKT;

        if (!epnum)

            ep->csr[0] &= ~MGC_M_CSR0_H_REQPKT;

#ifdef CLEAR_NAK

    }

#endif



    /* Clear all of the imaginable error bits first */

    ep->csr[1] &= ~(MGC_M_RXCSR_H_ERROR | MGC_M_RXCSR_H_RXSTALL |

                    MGC_M_RXCSR_DATAERROR);

    if (!epnum)

        ep->csr[0] &= ~(MGC_M_CSR0_H_ERROR | MGC_M_CSR0_H_RXSTALL |

                        MGC_M_CSR0_H_NAKTIMEOUT | MGC_M_CSR0_H_NO_PING);



    if (ep->status[1] == USB_RET_STALL) {

        ep->status[1] = 0;

        packey->len = 0;



        ep->csr[1] |= MGC_M_RXCSR_H_RXSTALL;

        if (!epnum)

            ep->csr[0] |= MGC_M_CSR0_H_RXSTALL;

    }



    if (ep->status[1] == USB_RET_NAK) {

        ep->status[1] = 0;



        /* NAK timeouts are only generated in Bulk transfers and

         * Data-errors in Isochronous.  */

        if (ep->interrupt[1])

            return musb_packet(s, ep, epnum, USB_TOKEN_IN,

                            packey->len, musb_rx_packet_complete, 1);



        ep->csr[1] |= MGC_M_RXCSR_DATAERROR;

        if (!epnum)

            ep->csr[0] |= MGC_M_CSR0_H_NAKTIMEOUT;

    }



    if (ep->status[1] < 0) {

        if (ep->status[1] == USB_RET_BABBLE) {

            musb_intr_set(s, musb_irq_rst_babble, 1);

            return;

        }



        /* Pretend we've tried three times already and failed (in

         * case of a control transfer).  */

        ep->csr[1] |= MGC_M_RXCSR_H_ERROR;

        if (!epnum)

            ep->csr[0] |= MGC_M_CSR0_H_ERROR;



        musb_rx_intr_set(s, epnum, 1);

        return;

    }

    /* TODO: check len for over/underruns of an OUT packet?  */

    /* TODO: perhaps make use of e->ext_size[1] here.  */



    packey->len = ep->status[1];



    if (!(ep->csr[1] & (MGC_M_RXCSR_H_RXSTALL | MGC_M_RXCSR_DATAERROR))) {

        ep->csr[1] |= MGC_M_RXCSR_FIFOFULL | MGC_M_RXCSR_RXPKTRDY;

        if (!epnum)

            ep->csr[0] |= MGC_M_CSR0_RXPKTRDY;



        ep->rxcount = packey->len; /* XXX: MIN(packey->len, ep->maxp[1]); */

        /* In DMA mode: assert DMA request for this EP */

    }



    /* Only if DMA has not been asserted */

    musb_rx_intr_set(s, epnum, 1);

}
