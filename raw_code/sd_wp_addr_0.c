static inline int sd_wp_addr(SDState *sd, uint32_t addr)

{

    return sd->wp_groups[addr >>

            (HWBLOCK_SHIFT + SECTOR_SHIFT + WPGROUP_SHIFT)];

}
