static bool vring_notify(VirtIODevice *vdev, VirtQueue *vq)

{

    uint16_t old, new;

    bool v;

    /* We need to expose used array entries before checking used event. */

    smp_mb();

    /* Always notify when queue is empty (when feature acknowledge) */

    if (virtio_has_feature(vdev, VIRTIO_F_NOTIFY_ON_EMPTY) &&

        !vq->inuse && vring_avail_idx(vq) == vq->last_avail_idx) {

        return true;

    }



    if (!virtio_has_feature(vdev, VIRTIO_RING_F_EVENT_IDX)) {

        return !(vring_avail_flags(vq) & VRING_AVAIL_F_NO_INTERRUPT);

    }



    v = vq->signalled_used_valid;

    vq->signalled_used_valid = true;

    old = vq->signalled_used;

    new = vq->signalled_used = vring_used_idx(vq);

    return !v || vring_need_event(vring_get_used_event(vq), new, old);

}
