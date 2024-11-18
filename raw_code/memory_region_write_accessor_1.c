static void memory_region_write_accessor(MemoryRegion *mr,

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

    tmp = (*value >> shift) & mask;

    trace_memory_region_ops_write(mr, addr, tmp, size);

    mr->ops->write(mr->opaque, addr, tmp, size);

}
