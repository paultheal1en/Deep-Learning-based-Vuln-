static void vfio_listener_region_del(MemoryListener *listener,

                                     MemoryRegionSection *section)

{

    VFIOContainer *container = container_of(listener, VFIOContainer, listener);

    hwaddr iova, end;

    Int128 llend, llsize;

    int ret;



    if (vfio_listener_skipped_section(section)) {

        trace_vfio_listener_region_del_skip(

                section->offset_within_address_space,

                section->offset_within_address_space +

                int128_get64(int128_sub(section->size, int128_one())));

        return;

    }



    if (unlikely((section->offset_within_address_space & ~TARGET_PAGE_MASK) !=

                 (section->offset_within_region & ~TARGET_PAGE_MASK))) {

        error_report("%s received unaligned region", __func__);

        return;

    }



    if (memory_region_is_iommu(section->mr)) {

        VFIOGuestIOMMU *giommu;



        QLIST_FOREACH(giommu, &container->giommu_list, giommu_next) {

            if (giommu->iommu == section->mr) {

                memory_region_unregister_iommu_notifier(giommu->iommu,

                                                        &giommu->n);

                QLIST_REMOVE(giommu, giommu_next);

                g_free(giommu);

                break;

            }

        }



        /*

         * FIXME: We assume the one big unmap below is adequate to

         * remove any individual page mappings in the IOMMU which

         * might have been copied into VFIO. This works for a page table

         * based IOMMU where a big unmap flattens a large range of IO-PTEs.

         * That may not be true for all IOMMU types.

         */

    }



    iova = TARGET_PAGE_ALIGN(section->offset_within_address_space);

    llend = int128_make64(section->offset_within_address_space);

    llend = int128_add(llend, section->size);

    llend = int128_and(llend, int128_exts64(TARGET_PAGE_MASK));



    if (int128_ge(int128_make64(iova), llend)) {

        return;

    }

    end = int128_get64(int128_sub(llend, int128_one()));



    llsize = int128_sub(llend, int128_make64(iova));



    trace_vfio_listener_region_del(iova, end);



    ret = vfio_dma_unmap(container, iova, int128_get64(llsize));

    memory_region_unref(section->mr);

    if (ret) {

        error_report("vfio_dma_unmap(%p, 0x%"HWADDR_PRIx", "

                     "0x%"HWADDR_PRIx") = %d (%m)",

                     container, iova, int128_get64(llsize), ret);

    }



    if (container->iommu_type == VFIO_SPAPR_TCE_v2_IOMMU) {

        vfio_spapr_remove_window(container,

                                 section->offset_within_address_space);

        if (vfio_host_win_del(container,

                              section->offset_within_address_space,

                              section->offset_within_address_space +

                              int128_get64(section->size) - 1) < 0) {

            hw_error("%s: Cannot delete missing window at %"HWADDR_PRIx,

                     __func__, section->offset_within_address_space);

        }

    }

}
