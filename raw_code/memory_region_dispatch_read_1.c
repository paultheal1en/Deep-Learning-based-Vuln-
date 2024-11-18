static bool memory_region_dispatch_read(MemoryRegion *mr,

                                        hwaddr addr,

                                        uint64_t *pval,

                                        unsigned size)

{

    if (!memory_region_access_valid(mr, addr, size, false)) {

        *pval = unassigned_mem_read(mr, addr, size);

        return true;

    }



    *pval = memory_region_dispatch_read1(mr, addr, size);

    adjust_endianness(mr, pval, size);

    return false;

}
