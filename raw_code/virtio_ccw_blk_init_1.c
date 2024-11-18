static int virtio_ccw_blk_init(VirtioCcwDevice *ccw_dev)

{

    VirtIOBlkCcw *dev = VIRTIO_BLK_CCW(ccw_dev);

    DeviceState *vdev = DEVICE(&dev->vdev);

    virtio_blk_set_conf(vdev, &(dev->blk));

    qdev_set_parent_bus(vdev, BUS(&ccw_dev->bus));

    if (qdev_init(vdev) < 0) {

        return -1;

    }



    return virtio_ccw_device_init(ccw_dev, VIRTIO_DEVICE(vdev));

}
