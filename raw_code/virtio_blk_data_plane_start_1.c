void virtio_blk_data_plane_start(VirtIOBlockDataPlane *s)

{

    BusState *qbus = BUS(qdev_get_parent_bus(DEVICE(s->vdev)));

    VirtioBusClass *k = VIRTIO_BUS_GET_CLASS(qbus);

    VirtIOBlock *vblk = VIRTIO_BLK(s->vdev);

    int r;



    if (vblk->dataplane_started || s->starting) {

        return;

    }



    s->starting = true;

    s->vq = virtio_get_queue(s->vdev, 0);



    /* Set up guest notifier (irq) */

    r = k->set_guest_notifiers(qbus->parent, 1, true);

    if (r != 0) {

        fprintf(stderr, "virtio-blk failed to set guest notifier (%d), "

                "ensure -enable-kvm is set\n", r);

        goto fail_guest_notifiers;

    }

    s->guest_notifier = virtio_queue_get_guest_notifier(s->vq);



    /* Set up virtqueue notify */

    r = k->set_host_notifier(qbus->parent, 0, true);

    if (r != 0) {

        fprintf(stderr, "virtio-blk failed to set host notifier (%d)\n", r);

        goto fail_host_notifier;

    }



    s->starting = false;

    vblk->dataplane_started = true;

    trace_virtio_blk_data_plane_start(s);



    blk_set_aio_context(s->conf->conf.blk, s->ctx);



    /* Kick right away to begin processing requests already in vring */

    event_notifier_set(virtio_queue_get_host_notifier(s->vq));



    /* Get this show started by hooking up our callbacks */

    aio_context_acquire(s->ctx);


    virtio_queue_aio_set_host_notifier_handler(s->vq, s->ctx, true, true);

    aio_context_release(s->ctx);

    return;



  fail_host_notifier:

    k->set_guest_notifiers(qbus->parent, 1, false);

  fail_guest_notifiers:

    vblk->dataplane_disabled = true;

    s->starting = false;

    vblk->dataplane_started = true;

}