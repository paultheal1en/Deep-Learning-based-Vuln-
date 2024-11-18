void memory_region_add_subregion(MemoryRegion *mr,

                                 hwaddr offset,

                                 MemoryRegion *subregion)

{

    subregion->may_overlap = false;

    subregion->priority = 0;

    memory_region_add_subregion_common(mr, offset, subregion);

}
