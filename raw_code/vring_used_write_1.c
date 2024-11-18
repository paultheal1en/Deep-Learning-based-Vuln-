static inline void vring_used_write(VirtQueue *vq, VRingUsedElem *uelem,

                                    int i)

{

    VRingMemoryRegionCaches *caches = atomic_rcu_read(&vq->vring.caches);

    hwaddr pa = offsetof(VRingUsed, ring[i]);

    virtio_tswap32s(vq->vdev, &uelem->id);

    virtio_tswap32s(vq->vdev, &uelem->len);

    address_space_write_cached(&caches->used, pa, uelem, sizeof(VRingUsedElem));

    address_space_cache_invalidate(&caches->used, pa, sizeof(VRingUsedElem));

}
