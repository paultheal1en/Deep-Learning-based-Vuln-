static int ehci_state_fetchsitd(EHCIState *ehci, int async)

{

    uint32_t entry;

    EHCIsitd sitd;



    assert(!async);

    entry = ehci_get_fetch_addr(ehci, async);



    get_dwords(NLPTR_GET(entry), (uint32_t *)&sitd,

               sizeof(EHCIsitd) >> 2);

    ehci_trace_sitd(ehci, entry, &sitd);



    if (!(sitd.results & SITD_RESULTS_ACTIVE)) {

        /* siTD is not active, nothing to do */;

    } else {

        /* TODO: split transfers are not implemented */

        fprintf(stderr, "WARNING: Skipping active siTD\n");

    }



    ehci_set_fetch_addr(ehci, async, sitd.next);

    ehci_set_state(ehci, async, EST_FETCHENTRY);

    return 1;

}
