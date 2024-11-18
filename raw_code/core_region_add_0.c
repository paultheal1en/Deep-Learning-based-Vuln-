static void core_region_add(MemoryListener *listener,

                            MemoryRegionSection *section)

{

    cpu_register_physical_memory_log(section, section->readonly);

}
