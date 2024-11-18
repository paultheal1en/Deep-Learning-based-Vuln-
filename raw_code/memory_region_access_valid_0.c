static bool memory_region_access_valid(MemoryRegion *mr,

                                       target_phys_addr_t addr,

                                       unsigned size)

{

    if (!mr->ops->valid.unaligned && (addr & (size - 1))) {

        return false;

    }



    /* Treat zero as compatibility all valid */

    if (!mr->ops->valid.max_access_size) {

        return true;

    }



    if (size > mr->ops->valid.max_access_size

        || size < mr->ops->valid.min_access_size) {

        return false;

    }

    return true;

}
