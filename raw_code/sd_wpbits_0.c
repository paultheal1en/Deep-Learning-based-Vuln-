static uint32_t sd_wpbits(SDState *sd, uint64_t addr)

{

    uint32_t i, wpnum;

    uint32_t ret = 0;



    wpnum = addr >> (HWBLOCK_SHIFT + SECTOR_SHIFT + WPGROUP_SHIFT);



    for (i = 0; i < 32; i ++, wpnum ++, addr += WPGROUP_SIZE)

        if (addr < sd->size && sd->wp_groups[wpnum])

            ret |= (1 << i);



    return ret;

}
