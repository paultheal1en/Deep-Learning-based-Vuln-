static void vtd_iommu_replay(MemoryRegion *mr, IOMMUNotifier *n)

{

    VTDAddressSpace *vtd_as = container_of(mr, VTDAddressSpace, iommu);

    IntelIOMMUState *s = vtd_as->iommu_state;

    uint8_t bus_n = pci_bus_num(vtd_as->bus);

    VTDContextEntry ce;



    if (vtd_dev_to_context_entry(s, bus_n, vtd_as->devfn, &ce) == 0) {

        /*

         * Scanned a valid context entry, walk over the pages and

         * notify when needed.

         */

        trace_vtd_replay_ce_valid(bus_n, PCI_SLOT(vtd_as->devfn),

                                  PCI_FUNC(vtd_as->devfn),

                                  VTD_CONTEXT_ENTRY_DID(ce.hi),

                                  ce.hi, ce.lo);

        vtd_page_walk(&ce, 0, ~0ULL, vtd_replay_hook, (void *)n);

    } else {

        trace_vtd_replay_ce_invalid(bus_n, PCI_SLOT(vtd_as->devfn),

                                    PCI_FUNC(vtd_as->devfn));

    }



    return;

}
