static void memory_region_destructor_alias(MemoryRegion *mr)

{

    memory_region_unref(mr->alias);

}
