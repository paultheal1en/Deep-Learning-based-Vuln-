static inline void vring_used_write(VirtQueue *vq, VRingUsedElem *uelem,

                                    int i)

{

    hwaddr pa;

    virtio_tswap32s(vq->vdev, &uelem->id);

    virtio_tswap32s(vq->vdev, &uelem->len);

    pa = vq->vring.used + offsetof(VRingUsed, ring[i]);

    address_space_write(&address_space_memory, pa, MEMTXATTRS_UNSPECIFIED,

                       (void *)uelem, sizeof(VRingUsedElem));

}
