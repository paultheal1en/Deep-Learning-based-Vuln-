static void virtio_ccw_post_plugged(DeviceState *d, Error **errp)

{

   VirtioCcwDevice *dev = VIRTIO_CCW_DEVICE(d);

   VirtIODevice *vdev = virtio_bus_get_device(&dev->bus);



   if (!virtio_host_has_feature(vdev, VIRTIO_F_VERSION_1)) {

        /* A backend didn't support modern virtio. */

       dev->max_rev = 0;

   }

}
