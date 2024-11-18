static void virtio_scsi_hotplug(HotplugHandler *hotplug_dev, DeviceState *dev,

                                Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(hotplug_dev);

    VirtIOSCSI *s = VIRTIO_SCSI(vdev);

    SCSIDevice *sd = SCSI_DEVICE(dev);



    if (s->ctx && !s->dataplane_disabled) {

        VirtIOSCSIBlkChangeNotifier *insert_notifier, *remove_notifier;



        if (blk_op_is_blocked(sd->conf.blk, BLOCK_OP_TYPE_DATAPLANE, errp)) {

            return;

        }

        blk_op_block_all(sd->conf.blk, s->blocker);

        aio_context_acquire(s->ctx);

        blk_set_aio_context(sd->conf.blk, s->ctx);

        aio_context_release(s->ctx);



        insert_notifier = g_new0(VirtIOSCSIBlkChangeNotifier, 1);

        insert_notifier->n.notify = virtio_scsi_blk_insert_notifier;

        insert_notifier->s = s;

        insert_notifier->sd = sd;

        blk_add_insert_bs_notifier(sd->conf.blk, &insert_notifier->n);

        QTAILQ_INSERT_TAIL(&s->insert_notifiers, insert_notifier, next);



        remove_notifier = g_new0(VirtIOSCSIBlkChangeNotifier, 1);

        remove_notifier->n.notify = virtio_scsi_blk_remove_notifier;

        remove_notifier->s = s;

        remove_notifier->sd = sd;

        blk_add_remove_bs_notifier(sd->conf.blk, &remove_notifier->n);

        QTAILQ_INSERT_TAIL(&s->remove_notifiers, remove_notifier, next);

    }



    if (virtio_vdev_has_feature(vdev, VIRTIO_SCSI_F_HOTPLUG)) {

        virtio_scsi_push_event(s, sd,

                               VIRTIO_SCSI_T_TRANSPORT_RESET,

                               VIRTIO_SCSI_EVT_RESET_RESCAN);

    }

}
