static MemTxResult memory_region_oldmmio_write_accessor(MemoryRegion *mr,

                                                        hwaddr addr,

                                                        uint64_t *value,

                                                        unsigned size,

                                                        unsigned shift,

                                                        uint64_t mask,

                                                        MemTxAttrs attrs)

{

    uint64_t tmp;



    tmp = (*value >> shift) & mask;

    if (mr->subpage) {

        trace_memory_region_subpage_write(get_cpu_index(), mr, addr, tmp, size);






    } else if (TRACE_MEMORY_REGION_OPS_WRITE_ENABLED) {

        hwaddr abs_addr = memory_region_to_absolute_addr(mr, addr);

        trace_memory_region_ops_write(get_cpu_index(), mr, abs_addr, tmp, size);

    }

    mr->ops->old_mmio.write[ctz32(size)](mr->opaque, addr, tmp);

    return MEMTX_OK;

}