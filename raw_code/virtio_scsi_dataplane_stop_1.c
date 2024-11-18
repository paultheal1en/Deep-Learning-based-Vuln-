void virtio_scsi_dataplane_stop(VirtIOSCSI *s)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIODevice *vdev = VIRTIO_DEVICE(s);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(s);

    int i;



    if (!s->dataplane_started || s->dataplane_stopping) {

        return;

    }

    error_free(s->blocker);

    s->blocker = NULL;

    s->dataplane_stopping = true;

    assert(s->ctx == iothread_get_aio_context(vs->conf.iothread));



    aio_context_acquire(s->ctx);



    aio_set_event_notifier(s->ctx, &s->ctrl_vring->host_notifier, NULL);

    aio_set_event_notifier(s->ctx, &s->event_vring->host_notifier, NULL);

    for (i = 0; i < vs->conf.num_queues; i++) {

        aio_set_event_notifier(s->ctx, &s->cmd_vrings[i]->host_notifier, NULL);

    }



    blk_drain_all(); /* ensure there are no in-flight requests */



    aio_context_release(s->ctx);



    /* Sync vring state back to virtqueue so that non-dataplane request

     * processing can continue when we disable the host notifier below.

     */

    vring_teardown(&s->ctrl_vring->vring, vdev, 0);

    vring_teardown(&s->event_vring->vring, vdev, 1);

    for (i = 0; i < vs->conf.num_queues; i++) {

        vring_teardown(&s->cmd_vrings[i]->vring, vdev, 2 + i);

    }



    for (i = 0; i < vs->conf.num_queues + 2; i++) {

        k->set_host_notifier(qbus->parent, i, false);

    }



    /* Clean up guest notifier (irq) */

    k->set_guest_notifiers(qbus->parent, vs->conf.num_queues + 2, false);

    s->dataplane_stopping = false;

    s->dataplane_started = false;

}
