static void vtd_reset_context_cache(IntelIOMMUState *s)

{

    VTDAddressSpace **pvtd_as;

    VTDAddressSpace *vtd_as;

    uint32_t bus_it;

    uint32_t devfn_it;



    VTD_DPRINTF(CACHE, "global context_cache_gen=1");

    for (bus_it = 0; bus_it < VTD_PCI_BUS_MAX; ++bus_it) {

        pvtd_as = s->address_spaces[bus_it];

        if (!pvtd_as) {

            continue;

        }

        for (devfn_it = 0; devfn_it < VTD_PCI_DEVFN_MAX; ++devfn_it) {

            vtd_as = pvtd_as[devfn_it];

            if (!vtd_as) {

                continue;

            }

            vtd_as->context_cache_entry.context_cache_gen = 0;

        }

    }

    s->context_cache_gen = 1;

}
