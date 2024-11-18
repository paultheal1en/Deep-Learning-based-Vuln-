static int t27(InterplayACMContext *s, unsigned ind, unsigned col)

{

    GetBitContext *gb = &s->gb;

    unsigned i, b;

    int n1, n2, n3;



    for (i = 0; i < s->rows; i++) {

        /* b = (x1) + (x2 * 5) + (x3 * 25) */

        b = get_bits(gb, 7);







        n1 =  (mul_3x5[b] & 0x0F) - 2;

        n2 = ((mul_3x5[b] >> 4) & 0x0F) - 2;

        n3 = ((mul_3x5[b] >> 8) & 0x0F) - 2;



        set_pos(s, i++, col, n1);

        if (i >= s->rows)

            break;

        set_pos(s, i++, col, n2);

        if (i >= s->rows)

            break;

        set_pos(s, i, col, n3);


    return 0;
