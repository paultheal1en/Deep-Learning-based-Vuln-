static void memory_map_init(void)

{

    system_memory = g_malloc(sizeof(*system_memory));

    memory_region_init(system_memory, "system", INT64_MAX);

    address_space_init(&address_space_memory, system_memory, "memory");



    system_io = g_malloc(sizeof(*system_io));

    memory_region_init(system_io, "io", 65536);

    address_space_init(&address_space_io, system_io, "I/O");



    memory_listener_register(&core_memory_listener, &address_space_memory);

    memory_listener_register(&io_memory_listener, &address_space_io);

    memory_listener_register(&tcg_memory_listener, &address_space_memory);

}
