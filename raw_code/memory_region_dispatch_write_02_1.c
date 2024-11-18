static bool memory_region_dispatch_write(MemoryRegion *mr,

                                         hwaddr addr,

                                         uint64_t data,

                                         unsigned size)

{

    if (!memory_region_access_valid(mr, addr, size, true)) {

        unassigned_mem_write(mr, addr, data, size);

        return true;

    }



    adjust_endianness(mr, &data, size);



    if (mr->ops->write) {

        access_with_adjusted_size(addr, &data, size,

                                  mr->ops->impl.min_access_size,

                                  mr->ops->impl.max_access_size,

                                  memory_region_write_accessor, mr);

    } else {

        access_with_adjusted_size(addr, &data, size, 1, 4,

                                  memory_region_oldmmio_write_accessor, mr);

    }

    return false;

}
