static void gen_store_exclusive(DisasContext *s, int rd, int rt, int rt2,

                                TCGv_i64 addr, int size, int is_pair)

{

    /* if (env->exclusive_addr == addr && env->exclusive_val == [addr]

     *     && (!is_pair || env->exclusive_high == [addr + datasize])) {

     *     [addr] = {Rt};

     *     if (is_pair) {

     *         [addr + datasize] = {Rt2};

     *     }

     *     {Rd} = 0;

     * } else {

     *     {Rd} = 1;

     * }

     * env->exclusive_addr = -1;

     */

    TCGLabel *fail_label = gen_new_label();

    TCGLabel *done_label = gen_new_label();

    TCGv_i64 tmp;



    tcg_gen_brcond_i64(TCG_COND_NE, addr, cpu_exclusive_addr, fail_label);



    tmp = tcg_temp_new_i64();

    if (is_pair) {

        if (size == 2) {

            if (s->be_data == MO_LE) {

                tcg_gen_concat32_i64(tmp, cpu_reg(s, rt), cpu_reg(s, rt2));

            } else {

                tcg_gen_concat32_i64(tmp, cpu_reg(s, rt2), cpu_reg(s, rt));

            }

            tcg_gen_atomic_cmpxchg_i64(tmp, cpu_exclusive_addr,

                                       cpu_exclusive_val, tmp,

                                       get_mem_index(s),

                                       MO_64 | MO_ALIGN | s->be_data);

            tcg_gen_setcond_i64(TCG_COND_NE, tmp, tmp, cpu_exclusive_val);

        } else if (s->be_data == MO_LE) {

            gen_helper_paired_cmpxchg64_le(tmp, cpu_env, cpu_exclusive_addr,

                                           cpu_reg(s, rt), cpu_reg(s, rt2));

        } else {

            gen_helper_paired_cmpxchg64_be(tmp, cpu_env, cpu_exclusive_addr,

                                           cpu_reg(s, rt), cpu_reg(s, rt2));

        }

    } else {

        tcg_gen_atomic_cmpxchg_i64(tmp, cpu_exclusive_addr, cpu_exclusive_val,

                                   cpu_reg(s, rt), get_mem_index(s),

                                   size | MO_ALIGN | s->be_data);

        tcg_gen_setcond_i64(TCG_COND_NE, tmp, tmp, cpu_exclusive_val);

    }

    tcg_gen_mov_i64(cpu_reg(s, rd), tmp);

    tcg_temp_free_i64(tmp);

    tcg_gen_br(done_label);



    gen_set_label(fail_label);

    tcg_gen_movi_i64(cpu_reg(s, rd), 1);

    gen_set_label(done_label);

    tcg_gen_movi_i64(cpu_exclusive_addr, -1);

}
