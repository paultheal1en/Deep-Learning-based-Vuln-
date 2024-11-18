static bool gen_wsr_ccount(DisasContext *dc, uint32_t sr, TCGv_i32 v)

{

    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_start();

    }

    gen_helper_wsr_ccount(cpu_env, v);

    if (dc->tb->cflags & CF_USE_ICOUNT) {

        gen_io_end();

        gen_jumpi_check_loop_end(dc, 0);

        return true;

    }

    return false;

}
