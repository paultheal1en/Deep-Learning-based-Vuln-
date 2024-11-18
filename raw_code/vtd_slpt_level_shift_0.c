static inline uint32_t vtd_slpt_level_shift(uint32_t level)

{

    return VTD_PAGE_SHIFT_4K + (level - 1) * VTD_SL_LEVEL_BITS;

}
