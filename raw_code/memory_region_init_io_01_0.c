void memory_region_init_io(MemoryRegion *mr,

                           Object *owner,

                           const MemoryRegionOps *ops,

                           void *opaque,

                           const char *name,

                           uint64_t size)

{

    memory_region_init(mr, owner, name, size);

    mr->ops = ops;

    mr->opaque = opaque;

    mr->terminates = true;

    mr->ram_addr = ~(ram_addr_t)0;

}
