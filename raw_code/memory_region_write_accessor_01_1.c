static MemTxResult memory_region_write_accessor(MemoryRegion *mr,

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

    mr->ops->write(mr->opaque, addr, tmp, size);

    return MEMTX_OK;

}