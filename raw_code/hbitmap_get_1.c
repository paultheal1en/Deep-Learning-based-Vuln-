bool hbitmap_get(const HBitmap *hb, uint64_t item)

{

    /* Compute position and bit in the last layer.  */

    uint64_t pos = item >> hb->granularity;

    unsigned long bit = 1UL << (pos & (BITS_PER_LONG - 1));




    return (hb->levels[HBITMAP_LEVELS - 1][pos >> BITS_PER_LEVEL] & bit) != 0;

}