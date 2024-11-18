static void vhost_dev_unassign_memory(struct vhost_dev *dev,

                                      uint64_t start_addr,

                                      uint64_t size)

{

    int from, to, n = dev->mem->nregions;

    /* Track overlapping/split regions for sanity checking. */

    int overlap_start = 0, overlap_end = 0, overlap_middle = 0, split = 0;



    for (from = 0, to = 0; from < n; ++from, ++to) {

        struct vhost_memory_region *reg = dev->mem->regions + to;

        uint64_t reglast;

        uint64_t memlast;

        uint64_t change;



        /* clone old region */

        if (to != from) {

            memcpy(reg, dev->mem->regions + from, sizeof *reg);

        }



        /* No overlap is simple */

        if (!ranges_overlap(reg->guest_phys_addr, reg->memory_size,

                            start_addr, size)) {

            continue;

        }



        /* Split only happens if supplied region

         * is in the middle of an existing one. Thus it can not

         * overlap with any other existing region. */

        assert(!split);



        reglast = range_get_last(reg->guest_phys_addr, reg->memory_size);

        memlast = range_get_last(start_addr, size);



        /* Remove whole region */

        if (start_addr <= reg->guest_phys_addr && memlast >= reglast) {

            --dev->mem->nregions;

            --to;

            assert(to >= 0);

            ++overlap_middle;

            continue;

        }



        /* Shrink region */

        if (memlast >= reglast) {

            reg->memory_size = start_addr - reg->guest_phys_addr;

            assert(reg->memory_size);

            assert(!overlap_end);

            ++overlap_end;

            continue;

        }



        /* Shift region */

        if (start_addr <= reg->guest_phys_addr) {

            change = memlast + 1 - reg->guest_phys_addr;

            reg->memory_size -= change;

            reg->guest_phys_addr += change;

            reg->userspace_addr += change;

            assert(reg->memory_size);

            assert(!overlap_start);

            ++overlap_start;

            continue;

        }



        /* This only happens if supplied region

         * is in the middle of an existing one. Thus it can not

         * overlap with any other existing region. */

        assert(!overlap_start);

        assert(!overlap_end);

        assert(!overlap_middle);

        /* Split region: shrink first part, shift second part. */

        memcpy(dev->mem->regions + n, reg, sizeof *reg);

        reg->memory_size = start_addr - reg->guest_phys_addr;

        assert(reg->memory_size);

        change = memlast + 1 - reg->guest_phys_addr;

        reg = dev->mem->regions + n;

        reg->memory_size -= change;

        assert(reg->memory_size);

        reg->guest_phys_addr += change;

        reg->userspace_addr += change;

        /* Never add more than 1 region */

        assert(dev->mem->nregions == n);

        ++dev->mem->nregions;

        ++split;

    }

}
