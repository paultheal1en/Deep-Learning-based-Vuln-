static void vhost_iommu_region_del(MemoryListener *listener,

                                   MemoryRegionSection *section)

{

    struct vhost_dev *dev = container_of(listener, struct vhost_dev,

                                         iommu_listener);

    struct vhost_iommu *iommu;



    if (!memory_region_is_iommu(section->mr)) {

        return;

    }



    QLIST_FOREACH(iommu, &dev->iommu_list, iommu_next) {

        if (iommu->mr == section->mr) {

            memory_region_unregister_iommu_notifier(iommu->mr,

                                                    &iommu->n);

            QLIST_REMOVE(iommu, iommu_next);

            g_free(iommu);

            break;

        }

    }

}
