static void ehci_advance_async_state(EHCIState *ehci)

{

    const int async = 1;



    switch(ehci_get_state(ehci, async)) {

    case EST_INACTIVE:

        if (!(ehci->usbcmd & USBCMD_ASE)) {

            break;

        }

        ehci_set_usbsts(ehci, USBSTS_ASS);

        ehci_set_state(ehci, async, EST_ACTIVE);

        // No break, fall through to ACTIVE



    case EST_ACTIVE:

        if ( !(ehci->usbcmd & USBCMD_ASE)) {

            ehci_clear_usbsts(ehci, USBSTS_ASS);

            ehci_set_state(ehci, async, EST_INACTIVE);

            break;

        }



        /* If the doorbell is set, the guest wants to make a change to the

         * schedule. The host controller needs to release cached data.

         * (section 4.8.2)

         */

        if (ehci->usbcmd & USBCMD_IAAD) {

            DPRINTF("ASYNC: doorbell request acknowledged\n");

            ehci->usbcmd &= ~USBCMD_IAAD;

            ehci_set_interrupt(ehci, USBSTS_IAA);

            break;

        }



        /* make sure guest has acknowledged */

        /* TO-DO: is this really needed? */

        if (ehci->usbsts & USBSTS_IAA) {

            DPRINTF("IAA status bit still set.\n");

            break;

        }



        /* check that address register has been set */

        if (ehci->asynclistaddr == 0) {

            break;

        }



        ehci_set_state(ehci, async, EST_WAITLISTHEAD);

        ehci_advance_state(ehci, async);

        break;



    default:

        /* this should only be due to a developer mistake */

        fprintf(stderr, "ehci: Bad asynchronous state %d. "

                "Resetting to active\n", ehci->astate);

        assert(0);

    }

}
