static uint32_t memory_region_read_thunk_n(void *_mr,

                                           target_phys_addr_t addr,

                                           unsigned size)

{

    MemoryRegion *mr = _mr;

    uint64_t data = 0;



    if (!memory_region_access_valid(mr, addr, size)) {

        return -1U; /* FIXME: better signalling */

    }



    if (!mr->ops->read) {

        return mr->ops->old_mmio.read[bitops_ffsl(size)](mr->opaque, addr);

    }



    /* FIXME: support unaligned access */

    access_with_adjusted_size(addr + mr->offset, &data, size,

                              mr->ops->impl.min_access_size,

                              mr->ops->impl.max_access_size,

                              memory_region_read_accessor, mr);



    return data;

}
