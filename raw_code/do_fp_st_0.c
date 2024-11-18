static void do_fp_st(DisasContext *s, int srcidx, TCGv_i64 tcg_addr, int size)

{

    /* This writes the bottom N bits of a 128 bit wide vector to memory */

    TCGv_i64 tmp = tcg_temp_new_i64();

    tcg_gen_ld_i64(tmp, cpu_env, fp_reg_offset(srcidx, MO_64));

    if (size < 4) {

        tcg_gen_qemu_st_i64(tmp, tcg_addr, get_mem_index(s), MO_TE + size);

    } else {

        TCGv_i64 tcg_hiaddr = tcg_temp_new_i64();

        tcg_gen_qemu_st_i64(tmp, tcg_addr, get_mem_index(s), MO_TEQ);

        tcg_gen_qemu_st64(tmp, tcg_addr, get_mem_index(s));

        tcg_gen_ld_i64(tmp, cpu_env, fp_reg_hi_offset(srcidx));

        tcg_gen_addi_i64(tcg_hiaddr, tcg_addr, 8);

        tcg_gen_qemu_st_i64(tmp, tcg_hiaddr, get_mem_index(s), MO_TEQ);

        tcg_temp_free_i64(tcg_hiaddr);

    }



    tcg_temp_free_i64(tmp);

}
