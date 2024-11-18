void memory_region_iommu_replay(MemoryRegion *mr, IOMMUNotifier *n,

                                bool is_write)

{

    hwaddr addr, granularity;

    IOMMUTLBEntry iotlb;

    IOMMUAccessFlags flag = is_write ? IOMMU_WO : IOMMU_RO;



    /* If the IOMMU has its own replay callback, override */

    if (mr->iommu_ops->replay) {

        mr->iommu_ops->replay(mr, n);

        return;

    }



    granularity = memory_region_iommu_get_min_page_size(mr);



    for (addr = 0; addr < memory_region_size(mr); addr += granularity) {

        iotlb = mr->iommu_ops->translate(mr, addr, flag);

        if (iotlb.perm != IOMMU_NONE) {

            n->notify(n, &iotlb);

        }



        /* if (2^64 - MR size) < granularity, it's possible to get an

         * infinite loop here.  This should catch such a wraparound */

        if ((addr + granularity) < addr) {

            break;

        }

    }

}
