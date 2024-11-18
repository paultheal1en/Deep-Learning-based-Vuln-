static int vfio_connect_container(VFIOGroup *group, AddressSpace *as,

                                  Error **errp)

{

    VFIOContainer *container;

    int ret, fd;

    VFIOAddressSpace *space;



    space = vfio_get_address_space(as);



    QLIST_FOREACH(container, &space->containers, next) {

        if (!ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &container->fd)) {

            group->container = container;

            QLIST_INSERT_HEAD(&container->group_list, group, container_next);

            vfio_kvm_device_add_group(group);

            return 0;





    fd = qemu_open("/dev/vfio/vfio", O_RDWR);

    if (fd < 0) {

        error_setg_errno(errp, errno, "failed to open /dev/vfio/vfio");

        ret = -errno;

        goto put_space_exit;




    ret = ioctl(fd, VFIO_GET_API_VERSION);

    if (ret != VFIO_API_VERSION) {

        error_setg(errp, "supported vfio version: %d, "

                   "reported version: %d", VFIO_API_VERSION, ret);

        ret = -EINVAL;

        goto close_fd_exit;




    container = g_malloc0(sizeof(*container));

    container->space = space;

    container->fd = fd;

    QLIST_INIT(&container->giommu_list);

    QLIST_INIT(&container->hostwin_list);

    if (ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU) ||

        ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU)) {

        bool v2 = !!ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1v2_IOMMU);

        struct vfio_iommu_type1_info info;



        ret = ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &fd);


            error_setg_errno(errp, errno, "failed to set group container");

            ret = -errno;

            goto free_container_exit;




        container->iommu_type = v2 ? VFIO_TYPE1v2_IOMMU : VFIO_TYPE1_IOMMU;



            error_setg_errno(errp, errno, "failed to set iommu for container");

            ret = -errno;

            goto free_container_exit;




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


        vfio_host_win_add(container, 0, (hwaddr)-1, info.iova_pgsizes);

    } else if (ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_SPAPR_TCE_IOMMU) ||

               ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_SPAPR_TCE_v2_IOMMU)) {

        struct vfio_iommu_spapr_tce_info info;

        bool v2 = !!ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_SPAPR_TCE_v2_IOMMU);



        ret = ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &fd);


            error_setg_errno(errp, errno, "failed to set group container");

            ret = -errno;

            goto free_container_exit;


        container->iommu_type =

            v2 ? VFIO_SPAPR_TCE_v2_IOMMU : VFIO_SPAPR_TCE_IOMMU;








            error_setg_errno(errp, errno, "failed to set iommu for container");

            ret = -errno;

            goto free_container_exit;




        /*

         * The host kernel code implementing VFIO_IOMMU_DISABLE is called

         * when container fd is closed so we do not call it explicitly

         * in this file.

         */

        if (!v2) {

            ret = ioctl(fd, VFIO_IOMMU_ENABLE);


                error_setg_errno(errp, errno, "failed to enable container");

                ret = -errno;

                goto free_container_exit;


        } else {

            container->prereg_listener = vfio_prereg_listener;



            memory_listener_register(&container->prereg_listener,

                                     &address_space_memory);

            if (container->error) {

                memory_listener_unregister(&container->prereg_listener);

                ret = container->error;

                error_setg(errp,

                    "RAM memory listener initialization failed for container");

                goto free_container_exit;





        info.argsz = sizeof(info);

        ret = ioctl(fd, VFIO_IOMMU_SPAPR_TCE_GET_INFO, &info);


            error_setg_errno(errp, errno,

                             "VFIO_IOMMU_SPAPR_TCE_GET_INFO failed");

            ret = -errno;

            if (v2) {

                memory_listener_unregister(&container->prereg_listener);


            goto free_container_exit;




        if (v2) {

            /*

             * There is a default window in just created container.

             * To make region_add/del simpler, we better remove this

             * window now and let those iommu_listener callbacks

             * create/remove them when needed.

             */

            ret = vfio_spapr_remove_window(container, info.dma32_window_start);


                error_setg_errno(errp, -ret,

                                 "failed to remove existing window");

                goto free_container_exit;


        } else {

            /* The default table uses 4K pages */

            vfio_host_win_add(container, info.dma32_window_start,

                              info.dma32_window_start +

                              info.dma32_window_size - 1,

                              0x1000);


    } else {

        error_setg(errp, "No available IOMMU models");

        ret = -EINVAL;

        goto free_container_exit;




    vfio_kvm_device_add_group(group);



    QLIST_INIT(&container->group_list);

    QLIST_INSERT_HEAD(&space->containers, container, next);



    group->container = container;

    QLIST_INSERT_HEAD(&container->group_list, group, container_next);



    container->listener = vfio_memory_listener;



    memory_listener_register(&container->listener, container->space->as);



    if (container->error) {

        ret = container->error;

        error_setg_errno(errp, -ret,

                         "memory listener initialization failed for container");

        goto listener_release_exit;




    container->initialized = true;



    return 0;

listener_release_exit:

    QLIST_REMOVE(group, container_next);

    QLIST_REMOVE(container, next);

    vfio_kvm_device_del_group(group);

    vfio_listener_release(container);



free_container_exit:

    g_free(container);



close_fd_exit:

    close(fd);



put_space_exit:

    vfio_put_address_space(space);



    return ret;
