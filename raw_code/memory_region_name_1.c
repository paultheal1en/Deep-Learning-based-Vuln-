const char *memory_region_name(const MemoryRegion *mr)

{

    return object_get_canonical_path_component(OBJECT(mr));

}
