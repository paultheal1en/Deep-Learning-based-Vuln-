static bool memory_region_get_may_overlap(Object *obj, Error **errp)

{

    MemoryRegion *mr = MEMORY_REGION(obj);



    return mr->may_overlap;

}
