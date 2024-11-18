address_space_translate_for_iotlb(CPUState *cpu, int asidx, hwaddr addr,

                                  hwaddr *xlat, hwaddr *plen)

{

    MemoryRegionSection *section;

    AddressSpaceDispatch *d = cpu->cpu_ases[asidx].memory_dispatch;



    section = address_space_translate_internal(d, addr, xlat, plen, false);



    assert(!section->mr->iommu_ops);

    return section;

}
