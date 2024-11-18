void memory_region_allocate_system_memory(MemoryRegion *mr, Object *owner,
                                          const char *name,
                                          uint64_t ram_size)
{
    uint64_t addr = 0;
    int i;
    if (nb_numa_nodes == 0 || !have_memdevs) {
        allocate_system_memory_nonnuma(mr, owner, name, ram_size);
        return;
    memory_region_init(mr, owner, name, ram_size);
    for (i = 0; i < MAX_NODES; i++) {
        Error *local_err = NULL;
        uint64_t size = numa_info[i].node_mem;
        HostMemoryBackend *backend = numa_info[i].node_memdev;
        if (!backend) {
            continue;
        MemoryRegion *seg = host_memory_backend_get_memory(backend, &local_err);
        if (local_err) {
            qerror_report_err(local_err);
        memory_region_add_subregion(mr, addr, seg);
        vmstate_register_ram_global(seg);
        addr += size;