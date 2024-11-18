static void gen_window_check1(DisasContext *dc, unsigned r1)

{

    if (dc->tb->flags & XTENSA_TBFLAG_EXCM) {

        return;

    }

    if (option_enabled(dc, XTENSA_OPTION_WINDOWED_REGISTER) &&

            r1 / 4 > dc->used_window) {

        TCGv_i32 pc = tcg_const_i32(dc->pc);

        TCGv_i32 w = tcg_const_i32(r1 / 4);



        dc->used_window = r1 / 4;

        gen_advance_ccount(dc);

        gen_helper_window_check(cpu_env, pc, w);



        tcg_temp_free(w);

        tcg_temp_free(pc);

    }

}
