static int vfio_connect_container(VFIOGroup *group, AddressSpace *as)

{

    VFIOContainer *container;

    int ret, fd;

    VFIOAddressSpace *space;



    space = vfio_get_address_space(as);



    QLIST_FOREACH(container, &space->containers, next) {

        if (!ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &container->fd)) {

            group->container = container;

            QLIST_INSERT_HEAD(&container->group_list, group, container_next);

            return 0;

        }

    }



    fd = qemu_open("/dev/vfio/vfio", O_RDWR);

    if (fd < 0) {

        error_report("vfio: failed to open /dev/vfio/vfio: %m");

        ret = -errno;

        goto put_space_exit;

    }



    ret = ioctl(fd, VFIO_GET_API_VERSION);

    if (ret != VFIO_API_VERSION) {

        error_report("vfio: supported vfio version: %d, "

                     "reported version: %d", VFIO_API_VERSION, ret);

        ret = -EINVAL;

        goto close_fd_exit;

    }



    container = g_malloc0(sizeof(*container));

    container->space = space;

    container->fd = fd;

    if (ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU) ||

        ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU)) {

        bool v2 = !!ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU);

        struct vfio_iommu_type1_info info;



        ret = ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &fd);

        if (ret) {

            error_report("vfio: failed to set group container: %m");

            ret = -errno;

            goto free_container_exit;

        }



        container->iommu_type = v2 ? VFIO_TYPE1v2_IOMMU : VFIO_TYPE1_IOMMU;

        ret = ioctl(fd, VFIO_SET_IOMMU, container->iommu_type);

        if (ret) {

            error_report("vfio: failed to set iommu for container: %m");

            ret = -errno;

            goto free_container_exit;

        }



        /*

         * FIXME: This assumes that a Type1 IOMMU can map any 64-bit

         * IOVA whatsoever.  That's not actually true, but the current

         * kernel interface doesn't tell us what it can map, and the

         * existing Type1 IOMMUs generally support any IOVA we're

         * going to actually try in practice.

         */

        info.argsz = sizeof(info);

        ret = ioctl(fd, VFIO_IOMMU_GET_INFO, &info);

        /* Ignore errors */

        if (ret || !(info.flags & VFIO_IOMMU_INFO_PGSIZES)) {

            /* Assume 4k IOVA page size */

            info.iova_pgsizes = 4096;

        }

        vfio_host_win_add(container, 0, (hwaddr)-1, info.iova_pgsizes);

    } else if (ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_SPAPR_TCE_IOMMU) ||

               ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_SPAPR_TCE_v2_IOMMU)) {

        struct vfio_iommu_spapr_tce_info info;

        bool v2 = !!ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_SPAPR_TCE_v2_IOMMU);



        ret = ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &fd);

        if (ret) {

            error_report("vfio: failed to set group container: %m");

            ret = -errno;

            goto free_container_exit;

        }

        container->iommu_type =

            v2 ? VFIO_SPAPR_TCE_v2_IOMMU : VFIO_SPAPR_TCE_IOMMU;

        ret = ioctl(fd, VFIO_SET_IOMMU, container->iommu_type);

        if (ret) {

            error_report("vfio: failed to set iommu for container: %m");

            ret = -errno;

            goto free_container_exit;

        }



        /*

         * The host kernel code implementing VFIO_IOMMU_DISABLE is called

         * when container fd is closed so we do not call it explicitly

         * in this file.

         */

        if (!v2) {

            ret = ioctl(fd, VFIO_IOMMU_ENABLE);

            if (ret) {

                error_report("vfio: failed to enable container: %m");

                ret = -errno;

                goto free_container_exit;

            }

        } else {

            container->prereg_listener = vfio_prereg_listener;



            memory_listener_register(&container->prereg_listener,

                                     &address_space_memory);

            if (container->error) {

                memory_listener_unregister(&container->prereg_listener);

                error_report("vfio: RAM memory listener initialization failed for container");

                goto free_container_exit;

            }

        }



        /*

         * This only considers the host IOMMU's 32-bit window.  At

         * some point we need to add support for the optional 64-bit

         * window and dynamic windows

         */

        info.argsz = sizeof(info);

        ret = ioctl(fd, VFIO_IOMMU_SPAPR_TCE_GET_INFO, &info);

        if (ret) {

            error_report("vfio: VFIO_IOMMU_SPAPR_TCE_GET_INFO failed: %m");

            ret = -errno;

            if (v2) {

                memory_listener_unregister(&container->prereg_listener);

            }

            goto free_container_exit;

        }



        /* The default table uses 4K pages */

        vfio_host_win_add(container, info.dma32_window_start,

                          info.dma32_window_start +

                          info.dma32_window_size - 1,

                          0x1000);

    } else {

        error_report("vfio: No available IOMMU models");

        ret = -EINVAL;

        goto free_container_exit;

    }



    container->listener = vfio_memory_listener;



    memory_listener_register(&container->listener, container->space->as);



    if (container->error) {

        ret = container->error;

        error_report("vfio: memory listener initialization failed for container");

        goto listener_release_exit;

    }



    container->initialized = true;



    QLIST_INIT(&container->group_list);

    QLIST_INSERT_HEAD(&space->containers, container, next);



    group->container = container;

    QLIST_INSERT_HEAD(&container->group_list, group, container_next);



    return 0;

listener_release_exit:

    vfio_listener_release(container);



free_container_exit:

    g_free(container);



close_fd_exit:

    close(fd);



put_space_exit:

    vfio_put_address_space(space);



    return ret;

}
