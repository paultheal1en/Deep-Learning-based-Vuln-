static int vhost_verify_ring_mappings(struct vhost_dev *dev,

                                      uint64_t start_addr,

                                      uint64_t size)

{

    int i;

    for (i = 0; i < dev->nvqs; ++i) {

        struct vhost_virtqueue *vq = dev->vqs + i;

        hwaddr l;

        void *p;



        if (!ranges_overlap(start_addr, size, vq->ring_phys, vq->ring_size)) {

            continue;

        }

        l = vq->ring_size;

        p = cpu_physical_memory_map(vq->ring_phys, &l, 1);

        if (!p || l != vq->ring_size) {

            fprintf(stderr, "Unable to map ring buffer for ring %d\n", i);

            return -ENOMEM;

        }

        if (p != vq->ring) {

            fprintf(stderr, "Ring buffer relocated for ring %d\n", i);

            return -EBUSY;

        }

        cpu_physical_memory_unmap(p, l, 0, 0);

    }

    return 0;

}
