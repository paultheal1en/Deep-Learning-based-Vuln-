static void put_line(uint8_t *dst, int size, int width, const int *runs)

{

    PutBitContext pb;

    int run, mode = ~0, pix_left = width, run_idx = 0;



    init_put_bits(&pb, dst, size * 8);

    while (pix_left > 0) {

        run       = runs[run_idx++];

        mode      = ~mode;

        pix_left -= run;

        for (; run > 16; run -= 16)

            put_sbits(&pb, 16, mode);

        if (run)

            put_sbits(&pb, run, mode);

    }

    flush_put_bits(&pb);

}
