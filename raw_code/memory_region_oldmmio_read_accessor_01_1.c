static void memory_region_oldmmio_read_accessor(MemoryRegion *mr,

                                                hwaddr addr,

                                                uint64_t *value,

                                                unsigned size,

                                                unsigned shift,

                                                uint64_t mask)

{

    uint64_t tmp;



    tmp = mr->ops->old_mmio.read[ctz32(size)](mr->opaque, addr);

    trace_memory_region_ops_read(mr, addr, tmp, size);

    *value |= (tmp & mask) << shift;

}
