bool vhost_dev_query(struct vhost_dev *hdev, VirtIODevice *vdev)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(vdev)));

    VirtioBusState *vbus = VIRTIO_BUS(qbus);

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(vbus);



    return !k->query_guest_notifiers ||

           k->query_guest_notifiers(qbus->parent) ||

           hdev->force;

}
