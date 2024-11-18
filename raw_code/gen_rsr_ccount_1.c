static bool gen_rsr_ccount(DisasContext *dc, TCGv_i32 d, uint32_t sr)

{

    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_start();

    }

    gen_helper_update_ccount(cpu_env);

    tcg_gen_mov_i32(d, cpu_SR[sr]);

    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_end();

        return true;

    }

    return false;

}
