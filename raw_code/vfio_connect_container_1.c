static int vfio_connect_container(VFIOGroup *group)

{

    VFIOContainer *container;

    int ret, fd;



    if (group->container) {

        return 0;

    }



    QLIST_FOREACH(container, &container_list, next) {

        if (!ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &container->fd)) {

            group->container = container;

            QLIST_INSERT_HEAD(&container->group_list, group, container_next);

            return 0;

        }

    }



    fd = qemu_open("/dev/vfio/vfio", O_RDWR);

    if (fd < 0) {

        error_report("vfio: failed to open /dev/vfio/vfio: %m");

        return -errno;

    }



    ret = ioctl(fd, VFIO_GET_API_VERSION);

    if (ret != VFIO_API_VERSION) {

        error_report("vfio: supported vfio version: %d, "

                     "reported version: %d", VFIO_API_VERSION, ret);

        ret = -EINVAL;

        goto close_fd_exit;

    }



    container = g_malloc0(sizeof(*container));

    container->fd = fd;



    if (ioctl(fd, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU)) {

        ret = ioctl(group->fd, VFIO_GROUP_SET_CONTAINER, &fd);

        if (ret) {

            error_report("vfio: failed to set group container: %m");

            ret = -errno;

            goto free_container_exit;

        }



        ret = ioctl(fd, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU);

        if (ret) {

            error_report("vfio: failed to set iommu for container: %m");

            ret = -errno;

            goto free_container_exit;

        }



        container->iommu_data.type1.listener = vfio_memory_listener;

        container->iommu_data.release = vfio_listener_release;



        memory_listener_register(&container->iommu_data.type1.listener,

                                 &address_space_memory);



        if (container->iommu_data.type1.error) {

            ret = container->iommu_data.type1.error;

            error_report("vfio: memory listener initialization failed for container");

            goto listener_release_exit;

        }



        container->iommu_data.type1.initialized = true;



    } else {

        error_report("vfio: No available IOMMU models");

        ret = -EINVAL;

        goto free_container_exit;

    }



    QLIST_INIT(&container->group_list);

    QLIST_INSERT_HEAD(&container_list, container, next);



    group->container = container;

    QLIST_INSERT_HEAD(&container->group_list, group, container_next);



    return 0;



listener_release_exit:

    vfio_listener_release(container);



free_container_exit:

    g_free(container);



close_fd_exit:

    close(fd);



    return ret;

}
