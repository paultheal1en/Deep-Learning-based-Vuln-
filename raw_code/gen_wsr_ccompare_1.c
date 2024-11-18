static bool gen_wsr_ccompare(DisasContext *dc, uint32_t sr, TCGv_i32 v)

{

    uint32_t id = sr - CCOMPARE;

    bool ret = false;



    if (id < dc->config->nccompare) {

        uint32_t int_bit = 1 << dc->config->timerint[id];

        TCGv_i32 tmp = tcg_const_i32(id);



        tcg_gen_mov_i32(cpu_SR[sr], v);

        tcg_gen_andi_i32(cpu_SR[INTSET], cpu_SR[INTSET], ~int_bit);

        if (dc->tb->cflags & CF_USE_ICOUNT) {

            gen_io_start();

        }

        gen_helper_update_ccompare(cpu_env, tmp);

        if (dc->tb->cflags & CF_USE_ICOUNT) {

            gen_io_end();

            gen_jumpi_check_loop_end(dc, 0);

            ret = true;

        }

        tcg_temp_free(tmp);

    }

    return ret;

}
