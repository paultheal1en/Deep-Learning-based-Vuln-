static int vfio_get_device(VFIOGroup *group, const char *name,

                           VFIOPCIDevice *vdev)

{

    struct vfio_device_info dev_info = { .argsz = sizeof(dev_info) };

    struct vfio_region_info reg_info = { .argsz = sizeof(reg_info) };

    struct vfio_irq_info irq_info = { .argsz = sizeof(irq_info) };

    int ret, i;



    ret = ioctl(group->fd, VFIO_GROUP_GET_DEVICE_FD, name);

    if (ret < 0) {

        error_report("vfio: error getting device %s from group %d: %m",

                     name, group->groupid);

        error_printf("Verify all devices in group %d are bound to vfio-pci "

                     "or pci-stub and not already in use\n", group->groupid);

        return ret;

    }



    vdev->vbasedev.fd = ret;

    vdev->vbasedev.group = group;

    QLIST_INSERT_HEAD(&group->device_list, vdev, next);



    /* Sanity check device */

    ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_INFO, &dev_info);

    if (ret) {

        error_report("vfio: error getting device info: %m");

        goto error;

    }



    trace_vfio_get_device_irq(name, dev_info.flags,

                              dev_info.num_regions, dev_info.num_irqs);



    if (!(dev_info.flags & VFIO_DEVICE_FLAGS_PCI)) {

        error_report("vfio: Um, this isn't a PCI device");

        goto error;

    }



    vdev->reset_works = !!(dev_info.flags & VFIO_DEVICE_FLAGS_RESET);



    if (dev_info.num_regions < VFIO_PCI_CONFIG_REGION_INDEX + 1) {

        error_report("vfio: unexpected number of io regions %u",

                     dev_info.num_regions);

        goto error;

    }



    if (dev_info.num_irqs < VFIO_PCI_MSIX_IRQ_INDEX + 1) {

        error_report("vfio: unexpected number of irqs %u", dev_info.num_irqs);

        goto error;

    }



    for (i = VFIO_PCI_BAR0_REGION_INDEX; i < VFIO_PCI_ROM_REGION_INDEX; i++) {

        reg_info.index = i;



        ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_REGION_INFO, &reg_info);

        if (ret) {

            error_report("vfio: Error getting region %d info: %m", i);

            goto error;

        }



        trace_vfio_get_device_region(name, i,

                                     (unsigned long)reg_info.size,

                                     (unsigned long)reg_info.offset,

                                     (unsigned long)reg_info.flags);



        vdev->bars[i].flags = reg_info.flags;

        vdev->bars[i].size = reg_info.size;

        vdev->bars[i].fd_offset = reg_info.offset;

        vdev->bars[i].fd = vdev->vbasedev.fd;

        vdev->bars[i].nr = i;

        QLIST_INIT(&vdev->bars[i].quirks);

    }



    reg_info.index = VFIO_PCI_CONFIG_REGION_INDEX;



    ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_REGION_INFO, &reg_info);

    if (ret) {

        error_report("vfio: Error getting config info: %m");

        goto error;

    }



    trace_vfio_get_device_config(name, (unsigned long)reg_info.size,

                                 (unsigned long)reg_info.offset,

                                 (unsigned long)reg_info.flags);



    vdev->config_size = reg_info.size;

    if (vdev->config_size == PCI_CONFIG_SPACE_SIZE) {

        vdev->pdev.cap_present &= ~QEMU_PCI_CAP_EXPRESS;

    }

    vdev->config_offset = reg_info.offset;



    if ((vdev->features & VFIO_FEATURE_ENABLE_VGA) &&

        dev_info.num_regions > VFIO_PCI_VGA_REGION_INDEX) {

        struct vfio_region_info vga_info = {

            .argsz = sizeof(vga_info),

            .index = VFIO_PCI_VGA_REGION_INDEX,

         };



        ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_REGION_INFO, &vga_info);

        if (ret) {

            error_report(

                "vfio: Device does not support requested feature x-vga");

            goto error;

        }



        if (!(vga_info.flags & VFIO_REGION_INFO_FLAG_READ) ||

            !(vga_info.flags & VFIO_REGION_INFO_FLAG_WRITE) ||

            vga_info.size < 0xbffff + 1) {

            error_report("vfio: Unexpected VGA info, flags 0x%lx, size 0x%lx",

                         (unsigned long)vga_info.flags,

                         (unsigned long)vga_info.size);

            goto error;

        }



        vdev->vga.fd_offset = vga_info.offset;

        vdev->vga.fd = vdev->vbasedev.fd;



        vdev->vga.region[QEMU_PCI_VGA_MEM].offset = QEMU_PCI_VGA_MEM_BASE;

        vdev->vga.region[QEMU_PCI_VGA_MEM].nr = QEMU_PCI_VGA_MEM;

        QLIST_INIT(&vdev->vga.region[QEMU_PCI_VGA_MEM].quirks);



        vdev->vga.region[QEMU_PCI_VGA_IO_LO].offset = QEMU_PCI_VGA_IO_LO_BASE;

        vdev->vga.region[QEMU_PCI_VGA_IO_LO].nr = QEMU_PCI_VGA_IO_LO;

        QLIST_INIT(&vdev->vga.region[QEMU_PCI_VGA_IO_LO].quirks);



        vdev->vga.region[QEMU_PCI_VGA_IO_HI].offset = QEMU_PCI_VGA_IO_HI_BASE;

        vdev->vga.region[QEMU_PCI_VGA_IO_HI].nr = QEMU_PCI_VGA_IO_HI;

        QLIST_INIT(&vdev->vga.region[QEMU_PCI_VGA_IO_HI].quirks);



        vdev->has_vga = true;

    }

    irq_info.index = VFIO_PCI_ERR_IRQ_INDEX;



    ret = ioctl(vdev->vbasedev.fd, VFIO_DEVICE_GET_IRQ_INFO, &irq_info);

    if (ret) {

        /* This can fail for an old kernel or legacy PCI dev */

        trace_vfio_get_device_get_irq_info_failure();

        ret = 0;

    } else if (irq_info.count == 1) {

        vdev->pci_aer = true;

    } else {

        error_report("vfio: %04x:%02x:%02x.%x "

                     "Could not enable error recovery for the device",

                     vdev->host.domain, vdev->host.bus, vdev->host.slot,

                     vdev->host.function);

    }



error:

    if (ret) {

        QLIST_REMOVE(vdev, next);

        vdev->vbasedev.group = NULL;

        close(vdev->vbasedev.fd);

    }

    return ret;

}
