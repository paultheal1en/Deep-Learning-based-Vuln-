static int ehci_state_fetchentry(EHCIState *ehci, int async)

{

    int again = 0;

    uint32_t entry = ehci_get_fetch_addr(ehci, async);



    if (entry < 0x1000) {

        DPRINTF("fetchentry: entry invalid (0x%08x)\n", entry);

        ehci_set_state(ehci, async, EST_ACTIVE);

        goto out;

    }



    /* section 4.8, only QH in async schedule */

    if (async && (NLPTR_TYPE_GET(entry) != NLPTR_TYPE_QH)) {

        fprintf(stderr, "non queue head request in async schedule\n");

        return -1;

    }



    switch (NLPTR_TYPE_GET(entry)) {

    case NLPTR_TYPE_QH:

        ehci_set_state(ehci, async, EST_FETCHQH);

        again = 1;

        break;



    case NLPTR_TYPE_ITD:

        ehci_set_state(ehci, async, EST_FETCHITD);

        again = 1;

        break;



    case NLPTR_TYPE_STITD:

        ehci_set_state(ehci, async, EST_FETCHSITD);

        again = 1;

        break;



    default:

        /* TODO: handle FSTN type */

        fprintf(stderr, "FETCHENTRY: entry at %X is of type %d "

                "which is not supported yet\n", entry, NLPTR_TYPE_GET(entry));

        return -1;

    }



out:

    return again;

}
