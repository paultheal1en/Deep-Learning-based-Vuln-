static inline uint64_t vtd_get_slpte_addr(uint64_t slpte)

{

    return slpte & VTD_SL_PT_BASE_ADDR_MASK(VTD_HOST_ADDRESS_WIDTH);

}
