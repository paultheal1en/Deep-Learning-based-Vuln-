static int cow_find_streak(const uint8_t *bitmap, int value, int start, int nb_sectors)

{

    int streak_value = value ? 0xFF : 0;

    int last = MIN(start + nb_sectors, BITS_PER_BITMAP_SECTOR);

    int bitnum = start;

    while (bitnum < last) {

        if ((bitnum & 7) == 0 && bitmap[bitnum / 8] == streak_value) {

            bitnum += 8;

            continue;

        }

        if (cow_test_bit(bitnum, bitmap) == value) {

            bitnum++;

            continue;

        }

        break;

    }

    return MIN(bitnum, last) - start;

}
