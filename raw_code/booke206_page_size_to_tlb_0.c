static inline hwaddr booke206_page_size_to_tlb(uint64_t size)

{

    return (ffs(size >> 10) - 1) >> 1;

}
