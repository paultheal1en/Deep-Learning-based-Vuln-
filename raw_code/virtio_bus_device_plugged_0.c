void virtio_bus_device_plugged(VirtIODevice *vdev, Error **errp)

{

    DeviceState *qdev = DEVICE(vdev);

    BusState *qbus = BUS(qdev_get_parent_bus(qdev));

    VirtioBusState *bus = VIRTIO_BUS(qbus);

    VirtioBusClass *klass = VIRTIO_BUS_GET_CLASS(bus);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_GET_CLASS(vdev);



    DPRINTF("%s: plug device.\n", qbus->name);



    if (klass->device_plugged != NULL) {

        klass->device_plugged(qbus->parent, errp);

    }



    /* Get the features of the plugged device. */

    assert(vdc->get_features != NULL);

    vdev->host_features = vdc->get_features(vdev, vdev->host_features,

                                            errp);

    if (klass->post_plugged != NULL) {

        klass->post_plugged(qbus->parent, errp);

    }

}
