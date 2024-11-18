static void virtio_ccw_device_plugged(DeviceState *d, Error **errp)

{

    VirtioCcwDevice *dev = VIRTIO_CCW_DEVICE(d);

    VirtIODevice *vdev = virtio_bus_get_device(&dev->bus);

    CcwDevice *ccw_dev = CCW_DEVICE(d);

    SubchDev *sch = ccw_dev->sch;

    int n = virtio_get_num_queues(vdev);

    S390FLICState *flic = s390_get_flic();



    if (!virtio_has_feature(vdev->host_features, VIRTIO_F_VERSION_1)) {

        dev->max_rev = 0;

    }



    if (virtio_get_num_queues(vdev) > VIRTIO_CCW_QUEUE_MAX) {

        error_setg(errp, "The number of virtqueues %d "

                   "exceeds ccw limit %d", n,

                   VIRTIO_CCW_QUEUE_MAX);

        return;

    }

    if (virtio_get_num_queues(vdev) > flic->adapter_routes_max_batch) {

        error_setg(errp, "The number of virtqueues %d "

                   "exceeds flic adapter route limit %d", n,

                   flic->adapter_routes_max_batch);

        return;

    }



    sch->id.cu_model = virtio_bus_get_vdev_id(&dev->bus);





    css_generate_sch_crws(sch->cssid, sch->ssid, sch->schid,

                          d->hotplugged, 1);

}
