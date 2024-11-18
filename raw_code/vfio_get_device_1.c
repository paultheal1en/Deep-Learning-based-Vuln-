static int vfio_get_device(VFIOGroup *group, const char *name, VFIODevice *vdev)

{

    struct vfio_device_info dev_info = { .argsz = sizeof(dev_info) };

    struct vfio_region_info reg_info = { .argsz = sizeof(reg_info) };

    int ret, i;



    ret = ioctl(group->fd, VFIO_GROUP_GET_DEVICE_FD, name);

    if (ret < 0) {

        error_report("vfio: error getting device %s from group %d: %m\n",

                     name, group->groupid);

        error_report("Verify all devices in group %d are bound to vfio-pci "

                     "or pci-stub and not already in use\n", group->groupid);

        return ret;

    }



    vdev->fd = ret;

    vdev->group = group;

    QLIST_INSERT_HEAD(&group->device_list, vdev, next);



    /* Sanity check device */

    ret = ioctl(vdev->fd, VFIO_DEVICE_GET_INFO, &dev_info);

    if (ret) {

        error_report("vfio: error getting device info: %m\n");

        goto error;

    }



    DPRINTF("Device %s flags: %u, regions: %u, irgs: %u\n", name,

            dev_info.flags, dev_info.num_regions, dev_info.num_irqs);



    if (!(dev_info.flags & VFIO_DEVICE_FLAGS_PCI)) {

        error_report("vfio: Um, this isn't a PCI device\n");

        goto error;

    }



    vdev->reset_works = !!(dev_info.flags & VFIO_DEVICE_FLAGS_RESET);

    if (!vdev->reset_works) {

        error_report("Warning, device %s does not support reset\n", name);

    }



    if (dev_info.num_regions != VFIO_PCI_NUM_REGIONS) {

        error_report("vfio: unexpected number of io regions %u\n",

                     dev_info.num_regions);

        goto error;

    }



    if (dev_info.num_irqs != VFIO_PCI_NUM_IRQS) {

        error_report("vfio: unexpected number of irqs %u\n", dev_info.num_irqs);

        goto error;

    }



    for (i = VFIO_PCI_BAR0_REGION_INDEX; i < VFIO_PCI_ROM_REGION_INDEX; i++) {

        reg_info.index = i;



        ret = ioctl(vdev->fd, VFIO_DEVICE_GET_REGION_INFO, &reg_info);

        if (ret) {

            error_report("vfio: Error getting region %d info: %m\n", i);

            goto error;

        }



        DPRINTF("Device %s region %d:\n", name, i);

        DPRINTF("  size: 0x%lx, offset: 0x%lx, flags: 0x%lx\n",

                (unsigned long)reg_info.size, (unsigned long)reg_info.offset,

                (unsigned long)reg_info.flags);



        vdev->bars[i].flags = reg_info.flags;

        vdev->bars[i].size = reg_info.size;

        vdev->bars[i].fd_offset = reg_info.offset;

        vdev->bars[i].fd = vdev->fd;

        vdev->bars[i].nr = i;

    }



    reg_info.index = VFIO_PCI_ROM_REGION_INDEX;



    ret = ioctl(vdev->fd, VFIO_DEVICE_GET_REGION_INFO, &reg_info);

    if (ret) {

        error_report("vfio: Error getting ROM info: %m\n");

        goto error;

    }



    DPRINTF("Device %s ROM:\n", name);

    DPRINTF("  size: 0x%lx, offset: 0x%lx, flags: 0x%lx\n",

            (unsigned long)reg_info.size, (unsigned long)reg_info.offset,

            (unsigned long)reg_info.flags);



    vdev->rom_size = reg_info.size;

    vdev->rom_offset = reg_info.offset;



    reg_info.index = VFIO_PCI_CONFIG_REGION_INDEX;



    ret = ioctl(vdev->fd, VFIO_DEVICE_GET_REGION_INFO, &reg_info);

    if (ret) {

        error_report("vfio: Error getting config info: %m\n");

        goto error;

    }



    DPRINTF("Device %s config:\n", name);

    DPRINTF("  size: 0x%lx, offset: 0x%lx, flags: 0x%lx\n",

            (unsigned long)reg_info.size, (unsigned long)reg_info.offset,

            (unsigned long)reg_info.flags);



    vdev->config_size = reg_info.size;

    vdev->config_offset = reg_info.offset;



error:

    if (ret) {

        QLIST_REMOVE(vdev, next);

        vdev->group = NULL;

        close(vdev->fd);

    }

    return ret;

}
