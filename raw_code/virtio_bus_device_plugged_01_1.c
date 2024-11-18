void virtio_bus_device_plugged(VirtIODevice *vdev, Error **errp)

{

    DeviceState *qdev = DEVICE(vdev);

    BusState *qbus = BUS(qdev_get_parent_bus(qdev));

    VirtioBusState *bus = VIRTIO_BUS(qbus);

    VirtioBusClass *klass = VIRTIO_BUS_GET_CLASS(bus);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_GET_CLASS(vdev);

    bool has_iommu = virtio_host_has_feature(vdev, VIRTIO_F_IOMMU_PLATFORM);



    DPRINTF("%s: plug device.\n", qbus->name);



    if (klass->pre_plugged != NULL) {

        klass->pre_plugged(qbus->parent, errp);

    }



    /* Get the features of the plugged device. */

    assert(vdc->get_features != NULL);

    vdev->host_features = vdc->get_features(vdev, vdev->host_features,

                                            errp);



    if (klass->device_plugged != NULL) {

        klass->device_plugged(qbus->parent, errp);

    }



    if (klass->get_dma_as != NULL && has_iommu) {

        virtio_add_feature(&vdev->host_features, VIRTIO_F_IOMMU_PLATFORM);

        vdev->dma_as = klass->get_dma_as(qbus->parent);

    } else {

        vdev->dma_as = &address_space_memory;

    }

}
