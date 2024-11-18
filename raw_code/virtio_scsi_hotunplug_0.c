static void virtio_scsi_hotunplug(HotplugHandler *hotplug_dev, DeviceState *dev,

                                  Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(hotplug_dev);

    VirtIOSCSI *s = VIRTIO_SCSI(vdev);

    SCSIDevice *sd = SCSI_DEVICE(dev);



    if ((vdev->guest_features >> VIRTIO_SCSI_F_HOTPLUG) & 1) {

        virtio_scsi_push_event(s, sd,

                               VIRTIO_SCSI_T_TRANSPORT_RESET,

                               VIRTIO_SCSI_EVT_RESET_REMOVED);

    }



    if (s->ctx) {

        blk_op_unblock_all(sd->conf.blk, s->blocker);

    }

    qdev_simple_device_unplug_cb(hotplug_dev, dev, errp);

}
