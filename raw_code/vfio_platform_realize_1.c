static void vfio_platform_realize(DeviceState *dev, Error **errp)

{

    VFIOPlatformDevice *vdev = VFIO_PLATFORM_DEVICE(dev);

    SysBusDevice *sbdev = SYS_BUS_DEVICE(dev);

    VFIODevice *vbasedev = &vdev->vbasedev;

    VFIOINTp *intp;

    int i, ret;



    vbasedev->type = VFIO_DEVICE_TYPE_PLATFORM;

    vbasedev->ops = &vfio_platform_ops;



    trace_vfio_platform_realize(vbasedev->name, vdev->compat);



    ret = vfio_base_device_init(vbasedev);

    if (ret) {

        error_setg(errp, "vfio: vfio_base_device_init failed for %s",

                   vbasedev->name);

        return;

    }



    for (i = 0; i < vbasedev->num_regions; i++) {

        vfio_map_region(vdev, i);

        sysbus_init_mmio(sbdev, &vdev->regions[i]->mem);

    }



    QLIST_FOREACH(intp, &vdev->intp_list, next) {

        vfio_start_eventfd_injection(intp);

    }

}
