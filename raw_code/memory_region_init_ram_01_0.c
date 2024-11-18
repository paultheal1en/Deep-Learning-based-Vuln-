void memory_region_init_ram(MemoryRegion *mr,

                            const char *name,

                            uint64_t size)

{

    memory_region_init(mr, name, size);

    mr->ram = true;

    mr->terminates = true;

    mr->destructor = memory_region_destructor_ram;

    mr->ram_addr = qemu_ram_alloc(size, mr);

    mr->backend_registered = true;

}
