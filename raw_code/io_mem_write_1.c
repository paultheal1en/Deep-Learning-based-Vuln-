bool io_mem_write(MemoryRegion *mr, hwaddr addr,

                  uint64_t val, unsigned size)

{

    return memory_region_dispatch_write(mr, addr, val, size);

}
