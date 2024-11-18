static IOMMUTLBEntry vtd_iommu_translate(MemoryRegion *iommu, hwaddr addr,

                                         bool is_write)

{

    VTDAddressSpace *vtd_as = container_of(iommu, VTDAddressSpace, iommu);

    IntelIOMMUState *s = vtd_as->iommu_state;

    IOMMUTLBEntry ret = {

        .target_as = &address_space_memory,

        .iova = addr,

        .translated_addr = 0,

        .addr_mask = ~(hwaddr)0,

        .perm = IOMMU_NONE,

    };



    if (!s->dmar_enabled) {

        /* DMAR disabled, passthrough, use 4k-page*/

        ret.iova = addr & VTD_PAGE_MASK_4K;

        ret.translated_addr = addr & VTD_PAGE_MASK_4K;

        ret.addr_mask = ~VTD_PAGE_MASK_4K;

        ret.perm = IOMMU_RW;

        return ret;

    }



    vtd_do_iommu_translate(vtd_as, vtd_as->bus, vtd_as->devfn, addr,

                           is_write, &ret);

    VTD_DPRINTF(MMU,

                "bus %"PRIu8 " slot %"PRIu8 " func %"PRIu8 " devfn %"PRIu8

                " iova 0x%"PRIx64 " hpa 0x%"PRIx64, pci_bus_num(vtd_as->bus),

                VTD_PCI_SLOT(vtd_as->devfn), VTD_PCI_FUNC(vtd_as->devfn),

                vtd_as->devfn, addr, ret.translated_addr);

    return ret;

}
