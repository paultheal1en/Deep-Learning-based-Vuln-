static void vhost_virtqueue_stop(struct vhost_dev *dev,

                                    struct VirtIODevice *vdev,

                                    struct vhost_virtqueue *vq,

                                    unsigned idx)

{

    int vhost_vq_index = dev->vhost_ops->vhost_get_vq_index(dev, idx);

    struct vhost_vring_state state = {

        .index = vhost_vq_index,

    };

    int r;



    r = dev->vhost_ops->vhost_get_vring_base(dev, &state);

    if (r < 0) {

        fprintf(stderr, "vhost VQ %d ring restore failed: %d\n", idx, r);

        fflush(stderr);

    }

    virtio_queue_set_last_avail_idx(vdev, idx, state.num);

    virtio_queue_invalidate_signalled_used(vdev, idx);



    /* In the cross-endian case, we need to reset the vring endianness to

     * native as legacy devices expect so by default.

     */

    if (!virtio_vdev_has_feature(vdev, VIRTIO_F_VERSION_1) &&

        vhost_needs_vring_endian(vdev)) {

        r = vhost_virtqueue_set_vring_endian_legacy(dev,

                                                    !virtio_is_big_endian(vdev),

                                                    vhost_vq_index);

        if (r < 0) {

            error_report("failed to reset vring endianness");

        }

    }



    assert (r >= 0);

    cpu_physical_memory_unmap(vq->ring, virtio_queue_get_ring_size(vdev, idx),

                              0, virtio_queue_get_ring_size(vdev, idx));

    cpu_physical_memory_unmap(vq->used, virtio_queue_get_used_size(vdev, idx),

                              1, virtio_queue_get_used_size(vdev, idx));

    cpu_physical_memory_unmap(vq->avail, virtio_queue_get_avail_size(vdev, idx),

                              0, virtio_queue_get_avail_size(vdev, idx));

    cpu_physical_memory_unmap(vq->desc, virtio_queue_get_desc_size(vdev, idx),

                              0, virtio_queue_get_desc_size(vdev, idx));

}
