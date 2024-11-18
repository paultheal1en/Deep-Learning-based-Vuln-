VirtIODevice *virtio_scsi_init(DeviceState *dev, VirtIOSCSIConf *proxyconf)

{

    VirtIOSCSI *s;

    static int virtio_scsi_id;

    size_t sz;

    int i;



    sz = sizeof(VirtIOSCSI) + proxyconf->num_queues * sizeof(VirtQueue *);

    s = (VirtIOSCSI *)virtio_common_init("virtio-scsi", VIRTIO_ID_SCSI,

                                         sizeof(VirtIOSCSIConfig), sz);



    s->qdev = dev;

    s->conf = proxyconf;



    /* TODO set up vdev function pointers */

    s->vdev.get_config = virtio_scsi_get_config;

    s->vdev.set_config = virtio_scsi_set_config;

    s->vdev.get_features = virtio_scsi_get_features;

    s->vdev.reset = virtio_scsi_reset;



    s->ctrl_vq = virtio_add_queue(&s->vdev, VIRTIO_SCSI_VQ_SIZE,

                                   virtio_scsi_handle_ctrl);

    s->event_vq = virtio_add_queue(&s->vdev, VIRTIO_SCSI_VQ_SIZE,

                                   NULL);

    for (i = 0; i < s->conf->num_queues; i++) {

        s->cmd_vqs[i] = virtio_add_queue(&s->vdev, VIRTIO_SCSI_VQ_SIZE,

                                         virtio_scsi_handle_cmd);

    }



    scsi_bus_new(&s->bus, dev, &virtio_scsi_scsi_info);

    if (!dev->hotplugged) {

        scsi_bus_legacy_handle_cmdline(&s->bus);

    }



    register_savevm(dev, "virtio-scsi", virtio_scsi_id++, 1,

                    virtio_scsi_save, virtio_scsi_load, s);



    return &s->vdev;

}
