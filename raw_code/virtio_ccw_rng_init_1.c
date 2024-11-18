static int virtio_ccw_rng_init(VirtioCcwDevice *ccw_dev)

{

    VirtIORNGCcw *dev = VIRTIO_RNG_CCW(ccw_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);



    qdev_set_parent_bus(vdev, BUS(&ccw_dev->bus));

    if (qdev_init(vdev) < 0) {

        return -1;

    }



    object_property_set_link(OBJECT(dev),

                             OBJECT(dev->vdev.conf.default_backend), "rng",

                             NULL);



    return virtio_ccw_device_init(ccw_dev, VIRTIO_DEVICE(vdev));

}
