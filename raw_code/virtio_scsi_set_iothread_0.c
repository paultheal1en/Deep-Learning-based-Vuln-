void virtio_scsi_set_iothread(VirtIOSCSI *s, IOThread *iothread)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);



    assert(!s->ctx);

    s->ctx = iothread_get_aio_context(vs->conf.iothread);



    /* Don't try if transport does not support notifiers. */

    if (!k->set_guest_notifiers || !k->ioeventfd_assign) {

        fprintf(stderr, "virtio-scsi: Failed to set iothread "

                   "(transport does not support notifiers)");

        exit(1);

    }

}
