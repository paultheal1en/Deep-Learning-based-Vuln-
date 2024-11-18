static void decode_gain_info(GetBitContext *gb, int *gaininfo)

{

    int i, n;



    while (get_bits1(gb)) {

        /* NOTHING */

    }



    n = get_bits_count(gb) - 1;     // amount of elements*2 to update



    i = 0;

    while (n--) {

        int index = get_bits(gb, 3);

        int gain = get_bits1(gb) ? get_bits(gb, 4) - 7 : -1;



        while (i <= index)

            gaininfo[i++] = gain;

    }

    while (i <= 8)

        gaininfo[i++] = 0;

}
