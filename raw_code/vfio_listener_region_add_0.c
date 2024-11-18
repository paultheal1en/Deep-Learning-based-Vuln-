static void vfio_listener_region_add(MemoryListener *listener,

                                     MemoryRegionSection *section)

{

    VFIOContainer *container = container_of(listener, VFIOContainer, listener);

    hwaddr iova, end;

    Int128 llend, llsize;

    void *vaddr;

    int ret;

    VFIOHostDMAWindow *hostwin;

    bool hostwin_found;



    if (vfio_listener_skipped_section(section)) {

        trace_vfio_listener_region_add_skip(

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



    iova = TARGET_PAGE_ALIGN(section->offset_within_address_space);

    llend = int128_make64(section->offset_within_address_space);

    llend = int128_add(llend, section->size);

    llend = int128_and(llend, int128_exts64(TARGET_PAGE_MASK));



    if (int128_ge(int128_make64(iova), llend)) {

        return;

    }

    end = int128_get64(int128_sub(llend, int128_one()));



    if (container->iommu_type == VFIO_SPAPR_TCE_v2_IOMMU) {

        VFIOHostDMAWindow *hostwin;

        hwaddr pgsize = 0;



        /* For now intersections are not allowed, we may relax this later */

        QLIST_FOREACH(hostwin, &container->hostwin_list, hostwin_next) {

            if (ranges_overlap(hostwin->min_iova,

                               hostwin->max_iova - hostwin->min_iova + 1,

                               section->offset_within_address_space,

                               int128_get64(section->size))) {

                ret = -1;

                goto fail;

            }

        }



        ret = vfio_spapr_create_window(container, section, &pgsize);

        if (ret) {

            goto fail;

        }



        vfio_host_win_add(container, section->offset_within_address_space,

                          section->offset_within_address_space +

                          int128_get64(section->size) - 1, pgsize);

    }



    hostwin_found = false;

    QLIST_FOREACH(hostwin, &container->hostwin_list, hostwin_next) {

        if (hostwin->min_iova <= iova && end <= hostwin->max_iova) {

            hostwin_found = true;

            break;

        }

    }



    if (!hostwin_found) {

        error_report("vfio: IOMMU container %p can't map guest IOVA region"

                     " 0x%"HWADDR_PRIx"..0x%"HWADDR_PRIx,

                     container, iova, end);

        ret = -EFAULT;

        goto fail;

    }



    memory_region_ref(section->mr);



    if (memory_region_is_iommu(section->mr)) {

        VFIOGuestIOMMU *giommu;



        trace_vfio_listener_region_add_iommu(iova, end);

        /*

         * FIXME: For VFIO iommu types which have KVM acceleration to

         * avoid bouncing all map/unmaps through qemu this way, this

         * would be the right place to wire that up (tell the KVM

         * device emulation the VFIO iommu handles to use).

         */

        giommu = g_malloc0(sizeof(*giommu));

        giommu->iommu = section->mr;

        giommu->iommu_offset = section->offset_within_address_space -

                               section->offset_within_region;

        giommu->container = container;

        giommu->n.notify = vfio_iommu_map_notify;

        giommu->n.notifier_flags = IOMMU_NOTIFIER_ALL;

        QLIST_INSERT_HEAD(&container->giommu_list, giommu, giommu_next);



        memory_region_register_iommu_notifier(giommu->iommu, &giommu->n);

        memory_region_iommu_replay(giommu->iommu, &giommu->n, false);



        return;

    }



    /* Here we assume that memory_region_is_ram(section->mr)==true */



    vaddr = memory_region_get_ram_ptr(section->mr) +

            section->offset_within_region +

            (iova - section->offset_within_address_space);



    trace_vfio_listener_region_add_ram(iova, end, vaddr);



    llsize = int128_sub(llend, int128_make64(iova));



    ret = vfio_dma_map(container, iova, int128_get64(llsize),

                       vaddr, section->readonly);

    if (ret) {

        error_report("vfio_dma_map(%p, 0x%"HWADDR_PRIx", "

                     "0x%"HWADDR_PRIx", %p) = %d (%m)",

                     container, iova, int128_get64(llsize), vaddr, ret);

        goto fail;

    }



    return;



fail:

    /*

     * On the initfn path, store the first error in the container so we

     * can gracefully fail.  Runtime, there's not much we can do other

     * than throw a hardware error.

     */

    if (!container->initialized) {

        if (!container->error) {

            container->error = ret;

        }

    } else {

        hw_error("vfio: DMA mapping failed, unable to continue");

    }

}
