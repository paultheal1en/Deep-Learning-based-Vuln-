static inline bool rom_order_compare(Rom *rom, Rom *item)

{

    return (rom->as > item->as) ||

           (rom->as == item->as && rom->addr >= item->addr);

}
