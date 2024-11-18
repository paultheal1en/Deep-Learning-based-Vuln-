static void vhost_scsi_stop(VHostSCSI *s)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(s);

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(vdev)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    int ret = 0;



    if (!k->set_guest_notifiers) {

        ret = k->set_guest_notifiers(qbus->parent, s->dev.nvqs, false);

        if (ret < 0) {

                error_report("vhost guest notifier cleanup failed: %d\n", ret);

        }

    }

    assert(ret >= 0);



    vhost_scsi_clear_endpoint(s);

    vhost_dev_stop(&s->dev, vdev);

    vhost_dev_disable_notifiers(&s->dev, vdev);

}
