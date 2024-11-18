static bool vfio_listener_skipped_section(MemoryRegionSection *section)

{

    return !memory_region_is_ram(section->mr);

}
