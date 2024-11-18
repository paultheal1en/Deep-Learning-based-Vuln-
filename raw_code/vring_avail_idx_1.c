static inline uint16_t vring_avail_idx(VirtQueue *vq)

{

    VRingMemoryRegionCaches *caches = atomic_rcu_read(&vq->vring.caches);

    hwaddr pa = offsetof(VRingAvail, idx);

    vq->shadow_avail_idx = virtio_lduw_phys_cached(vq->vdev, &caches->avail, pa);

    return vq->shadow_avail_idx;

}
