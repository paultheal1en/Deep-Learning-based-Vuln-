MemoryRegion *iotlb_to_region(hwaddr index)

{

    return address_space_memory.dispatch->sections[index & ~TARGET_PAGE_MASK].mr;

}
