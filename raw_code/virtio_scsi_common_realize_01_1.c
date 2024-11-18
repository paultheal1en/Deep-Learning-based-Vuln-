void virtio_scsi_common_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtIOSCSICommon *s = VIRTIO_SCSI_COMMON(dev);

    int i;



    virtio_init(vdev, "virtio-scsi", VIRTIO_ID_SCSI,

                sizeof(VirtIOSCSIConfig));



    s->cmd_vqs = g_malloc0(s->conf.num_queues * sizeof(VirtQueue *));

    s->sense_size = VIRTIO_SCSI_SENSE_SIZE;

    s->cdb_size = VIRTIO_SCSI_CDB_SIZE;



    s->ctrl_vq = virtio_add_queue(vdev, VIRTIO_SCSI_VQ_SIZE,

                                  virtio_scsi_handle_ctrl);

    s->event_vq = virtio_add_queue(vdev, VIRTIO_SCSI_VQ_SIZE,

                                   virtio_scsi_handle_event);

    for (i = 0; i < s->conf.num_queues; i++) {

        s->cmd_vqs[i] = virtio_add_queue(vdev, VIRTIO_SCSI_VQ_SIZE,

                                         virtio_scsi_handle_cmd);

    }

}
