static inline void vring_set_avail_event(VirtQueue *vq, uint16_t val)

{

    VRingMemoryRegionCaches *caches;

    hwaddr pa;

    if (!vq->notification) {

        return;

    }



    caches = atomic_rcu_read(&vq->vring.caches);

    pa = offsetof(VRingUsed, ring[vq->vring.num]);

    virtio_stw_phys_cached(vq->vdev, &caches->used, pa, val);

    address_space_cache_invalidate(&caches->used, pa, sizeof(val));

}
