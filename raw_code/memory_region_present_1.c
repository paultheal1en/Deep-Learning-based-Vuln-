bool memory_region_present(MemoryRegion *container, hwaddr addr)

{

    MemoryRegion *mr = memory_region_find(container, addr, 1).mr;

    if (!mr || (mr == container)) {

        return false;

    }

    memory_region_unref(mr);

    return true;

}
