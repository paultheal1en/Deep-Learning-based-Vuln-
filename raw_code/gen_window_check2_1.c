static void gen_window_check2(DisasContext *dc, unsigned r1, unsigned r2)

{

    gen_window_check1(dc, r1 > r2 ? r1 : r2);

}
