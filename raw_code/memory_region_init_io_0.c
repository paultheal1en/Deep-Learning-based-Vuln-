void memory_region_init_io(MemoryRegion *mr,

                           const MemoryRegionOps *ops,

                           void *opaque,

                           const char *name,

                           uint64_t size)

{

    memory_region_init(mr, name, size);

    mr->ops = ops;

    mr->opaque = opaque;

    mr->terminates = true;

    mr->backend_registered = false;

}
