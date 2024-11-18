void memory_region_notify_iommu(MemoryRegion *mr,

                                IOMMUTLBEntry entry)

{

    IOMMUNotifier *iommu_notifier;

    IOMMUNotifierFlag request_flags;



    assert(memory_region_is_iommu(mr));



    if (entry.perm & IOMMU_RW) {

        request_flags = IOMMU_NOTIFIER_MAP;

    } else {

        request_flags = IOMMU_NOTIFIER_UNMAP;

    }



    IOMMU_NOTIFIER_FOREACH(iommu_notifier, mr) {

        /*

         * Skip the notification if the notification does not overlap

         * with registered range.

         */

        if (iommu_notifier->start > entry.iova + entry.addr_mask + 1 ||

            iommu_notifier->end < entry.iova) {

            continue;

        }

        if (iommu_notifier->notifier_flags & request_flags) {

            iommu_notifier->notify(iommu_notifier, &entry);

        }

    }

}
