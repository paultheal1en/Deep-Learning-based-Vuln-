static void ehci_advance_async_state(EHCIState *ehci)

{

    const int async = 1;



    switch(ehci_get_state(ehci, async)) {

    case EST_INACTIVE:

        if (!ehci_async_enabled(ehci)) {

            break;

        }

        ehci_set_state(ehci, async, EST_ACTIVE);

        // No break, fall through to ACTIVE



    case EST_ACTIVE:

        if (!ehci_async_enabled(ehci)) {

            ehci_queues_rip_all(ehci, async);

            ehci_set_state(ehci, async, EST_INACTIVE);

            break;

        }



        /* make sure guest has acknowledged the doorbell interrupt */

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



        /* If the doorbell is set, the guest wants to make a change to the

         * schedule. The host controller needs to release cached data.

         * (section 4.8.2)

         */

        if (ehci->usbcmd & USBCMD_IAAD) {

            /* Remove all unseen qhs from the async qhs queue */

            ehci_queues_rip_unused(ehci, async, 1);

            DPRINTF("ASYNC: doorbell request acknowledged\n");

            ehci->usbcmd &= ~USBCMD_IAAD;

            ehci_set_interrupt(ehci, USBSTS_IAA);

        }

        break;



    default:

        /* this should only be due to a developer mistake */

        fprintf(stderr, "ehci: Bad asynchronous state %d. "

                "Resetting to active\n", ehci->astate);

        assert(0);

    }

}
