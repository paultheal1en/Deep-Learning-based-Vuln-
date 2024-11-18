void memory_region_init_iommu(MemoryRegion *mr,

                              Object *owner,

                              const MemoryRegionIOMMUOps *ops,

                              const char *name,

                              uint64_t size)

{

    memory_region_init(mr, owner, name, size);

    mr->iommu_ops = ops,

    mr->terminates = true;  /* then re-forwards */

    notifier_list_init(&mr->iommu_notify);

}
