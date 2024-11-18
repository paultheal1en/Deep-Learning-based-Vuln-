static inline void tlb_reset_dirty_range(CPUTLBEntry *tlb_entry,

                                         unsigned long start, unsigned long length)

{

    unsigned long addr;

    if ((tlb_entry->addr_write & ~TARGET_PAGE_MASK) == io_mem_ram.ram_addr) {

        addr = (tlb_entry->addr_write & TARGET_PAGE_MASK) + tlb_entry->addend;

        if ((addr - start) < length) {

            tlb_entry->addr_write = (tlb_entry->addr_write & TARGET_PAGE_MASK) | TLB_NOTDIRTY;

        }

    }

}
