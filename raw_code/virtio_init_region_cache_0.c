static void virtio_init_region_cache(VirtIODevice *vdev, int n)

{

    VirtQueue *vq = &vdev->vq[n];

    VRingMemoryRegionCaches *old = vq->vring.caches;

    VRingMemoryRegionCaches *new;

    hwaddr addr, size;

    int event_size;



    event_size = virtio_vdev_has_feature(vq->vdev, VIRTIO_RING_F_EVENT_IDX) ? 2 : 0;



    addr = vq->vring.desc;

    if (!addr) {

        return;

    }

    new = g_new0(VRingMemoryRegionCaches, 1);

    size = virtio_queue_get_desc_size(vdev, n);

    address_space_cache_init(&new->desc, vdev->dma_as,

                             addr, size, false);



    size = virtio_queue_get_used_size(vdev, n) + event_size;

    address_space_cache_init(&new->used, vdev->dma_as,

                             vq->vring.used, size, true);



    size = virtio_queue_get_avail_size(vdev, n) + event_size;

    address_space_cache_init(&new->avail, vdev->dma_as,

                             vq->vring.avail, size, false);



    atomic_rcu_set(&vq->vring.caches, new);

    if (old) {

        call_rcu(old, virtio_free_region_cache, rcu);

    }

}
