static int disas_cp_insn(CPUState *env, DisasContext *s, uint32_t insn)

{

    TCGv tmp, tmp2;

    uint32_t rd = (insn >> 12) & 0xf;

    uint32_t cp = (insn >> 8) & 0xf;

    if (IS_USER(s)) {

        return 1;

    }



    if (insn & ARM_CP_RW_BIT) {

        if (!env->cp[cp].cp_read)

            return 1;

        gen_set_pc_im(s->pc);

        tmp = new_tmp();

        tmp2 = tcg_const_i32(insn);

        gen_helper_get_cp(tmp, cpu_env, tmp2);

        tcg_temp_free(tmp2);

        store_reg(s, rd, tmp);

    } else {

        if (!env->cp[cp].cp_write)

            return 1;

        gen_set_pc_im(s->pc);

        tmp = load_reg(s, rd);

        tmp2 = tcg_const_i32(insn);

        gen_helper_set_cp(cpu_env, tmp2, tmp);

        tcg_temp_free(tmp2);

        dead_tmp(tmp);

    }

    return 0;

}
