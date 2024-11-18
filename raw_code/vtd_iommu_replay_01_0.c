static void vtd_iommu_replay(IOMMUMemoryRegion *iommu_mr, IOMMUNotifier *n)

{

    VTDAddressSpace *vtd_as = container_of(iommu_mr, VTDAddressSpace, iommu);

    IntelIOMMUState *s = vtd_as->iommu_state;

    uint8_t bus_n = pci_bus_num(vtd_as->bus);

    VTDContextEntry ce;



    /*

     * The replay can be triggered by either a invalidation or a newly

     * created entry. No matter what, we release existing mappings

     * (it means flushing caches for UNMAP-only registers).

     */

    vtd_address_space_unmap(vtd_as, n);



    if (vtd_dev_to_context_entry(s, bus_n, vtd_as->devfn, &ce) == 0) {

        trace_vtd_replay_ce_valid(bus_n, PCI_SLOT(vtd_as->devfn),

                                  PCI_FUNC(vtd_as->devfn),

                                  VTD_CONTEXT_ENTRY_DID(ce.hi),

                                  ce.hi, ce.lo);

        vtd_page_walk(&ce, 0, ~0ULL, vtd_replay_hook, (void *)n, false);

    } else {

        trace_vtd_replay_ce_invalid(bus_n, PCI_SLOT(vtd_as->devfn),

                                    PCI_FUNC(vtd_as->devfn));

    }



    return;

}
