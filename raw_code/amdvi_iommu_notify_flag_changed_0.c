static void amdvi_iommu_notify_flag_changed(MemoryRegion *iommu,

                                            IOMMUNotifierFlag old,

                                            IOMMUNotifierFlag new)

{

    AMDVIAddressSpace *as = container_of(iommu, AMDVIAddressSpace, iommu);



    hw_error("device %02x.%02x.%x requires iommu notifier which is not "

             "currently supported", as->bus_num, PCI_SLOT(as->devfn),

             PCI_FUNC(as->devfn));

}
