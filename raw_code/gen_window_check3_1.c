static void gen_window_check3(DisasContext *dc, unsigned r1, unsigned r2,

        unsigned r3)

{

    gen_window_check2(dc, r1, r2 > r3 ? r2 : r3);

}
