void memory_region_init_rom_device(MemoryRegion *mr,

                                   const MemoryRegionOps *ops,

                                   void *opaque,

                                   const char *name,

                                   uint64_t size)

{

    memory_region_init(mr, name, size);

    mr->ops = ops;

    mr->opaque = opaque;

    mr->terminates = true;

    mr->destructor = memory_region_destructor_rom_device;

    mr->ram_addr = qemu_ram_alloc(size, mr);

    mr->ram_addr |= cpu_register_io_memory(memory_region_read_thunk,

                                           memory_region_write_thunk,

                                           mr);

    mr->ram_addr |= IO_MEM_ROMD;

    mr->backend_registered = true;

}
