bool vring_setup(Vring *vring, VirtIODevice *vdev, int n)

{

    hwaddr vring_addr = virtio_queue_get_ring_addr(vdev, n);

    hwaddr vring_size = virtio_queue_get_ring_size(vdev, n);

    void *vring_ptr;



    vring->broken = false;



    hostmem_init(&vring->hostmem);

    vring_ptr = hostmem_lookup(&vring->hostmem, vring_addr, vring_size, true);

    if (!vring_ptr) {

        error_report("Failed to map vring "

                     "addr %#" HWADDR_PRIx " size %" HWADDR_PRIu,

                     vring_addr, vring_size);

        vring->broken = true;

        return false;

    }



    vring_init(&vring->vr, virtio_queue_get_num(vdev, n), vring_ptr, 4096);



    vring->last_avail_idx = 0;

    vring->last_used_idx = 0;

    vring->signalled_used = 0;

    vring->signalled_used_valid = false;



    trace_vring_setup(virtio_queue_get_ring_addr(vdev, n),

                      vring->vr.desc, vring->vr.avail, vring->vr.used);

    return true;

}
