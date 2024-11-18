static inline uint16_t vring_avail_flags(VirtQueue *vq)

{

    VRingMemoryRegionCaches *caches = atomic_rcu_read(&vq->vring.caches);

    hwaddr pa = offsetof(VRingAvail, flags);

    return virtio_lduw_phys_cached(vq->vdev, &caches->avail, pa);

}
