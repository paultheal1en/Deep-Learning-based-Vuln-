void virtio_queue_update_rings(VirtIODevice *vdev, int n)

{

    VRing *vring = &vdev->vq[n].vring;



    if (!vring->desc) {

        /* not yet setup -> nothing to do */

        return;

    }

    vring->avail = vring->desc + vring->num * sizeof(VRingDesc);

    vring->used = vring_align(vring->avail +

                              offsetof(VRingAvail, ring[vring->num]),

                              vring->align);

    virtio_init_region_cache(vdev, n);

}
