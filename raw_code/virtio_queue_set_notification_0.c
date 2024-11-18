void virtio_queue_set_notification(VirtQueue *vq, int enable)

{

    vq->notification = enable;

    if (virtio_has_feature(vq->vdev, VIRTIO_RING_F_EVENT_IDX)) {

        vring_set_avail_event(vq, vring_avail_idx(vq));

    } else if (enable) {

        vring_used_flags_unset_bit(vq, VRING_USED_F_NO_NOTIFY);

    } else {

        vring_used_flags_set_bit(vq, VRING_USED_F_NO_NOTIFY);

    }

    if (enable) {

        /* Expose avail event/used flags before caller checks the avail idx. */

        smp_mb();

    }

}
