static void memory_region_read_accessor(MemoryRegion *mr,

                                        hwaddr addr,

                                        uint64_t *value,

                                        unsigned size,

                                        unsigned shift,

                                        uint64_t mask)

{

    uint64_t tmp;



    if (mr->flush_coalesced_mmio) {

        qemu_flush_coalesced_mmio_buffer();

    }

    tmp = mr->ops->read(mr->opaque, addr, size);

    trace_memory_region_ops_read(mr, addr, tmp, size);

    *value |= (tmp & mask) << shift;

}
