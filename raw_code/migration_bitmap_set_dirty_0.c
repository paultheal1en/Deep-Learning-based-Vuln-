static inline void migration_bitmap_set_dirty(MemoryRegion *mr, int length)

{

    ram_addr_t addr;



    for (addr = 0; addr < length; addr += TARGET_PAGE_SIZE) {

        if (!memory_region_get_dirty(mr, addr, TARGET_PAGE_SIZE,

                                     DIRTY_MEMORY_MIGRATION)) {

            memory_region_set_dirty(mr, addr, TARGET_PAGE_SIZE);

        }

    }

}
