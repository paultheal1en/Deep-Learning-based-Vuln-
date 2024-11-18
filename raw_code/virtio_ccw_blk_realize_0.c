static void virtio_ccw_blk_realize(VirtioCcwDevice *ccw_dev, Error **errp)

{

    VirtIOBlkCcw *dev = VIRTIO_BLK_CCW(ccw_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);

    Error *err = NULL;



    qdev_set_parent_bus(vdev, BUS(&ccw_dev->bus));

    object_property_set_bool(OBJECT(vdev), true, "realized", &err);

    if (err) {

        error_propagate(errp, err);

    }

}
