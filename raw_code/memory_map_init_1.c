static void memory_map_init(void)

{

    system_memory = qemu_malloc(sizeof(*system_memory));

    memory_region_init(system_memory, "system", UINT64_MAX);

    set_system_memory_map(system_memory);

}
