static inline void gen_op_mulscc(TCGv dst, TCGv src1, TCGv src2)

{

    TCGv r_temp, zero;



    r_temp = tcg_temp_new();



    /* old op:

    if (!(env->y & 1))

        T1 = 0;

    */

    zero = tcg_const_tl(0);

    tcg_gen_andi_tl(cpu_cc_src, src1, 0xffffffff);

    tcg_gen_andi_tl(r_temp, cpu_y, 0x1);

    tcg_gen_andi_tl(cpu_cc_src2, src2, 0xffffffff);

    tcg_gen_movcond_tl(TCG_COND_EQ, cpu_cc_src2, r_temp, zero,

                       zero, cpu_cc_src2);

    tcg_temp_free(zero);



    // b2 = T0 & 1;

    // env->y = (b2 << 31) | (env->y >> 1);

    tcg_gen_andi_tl(r_temp, cpu_cc_src, 0x1);

    tcg_gen_shli_tl(r_temp, r_temp, 31);

    tcg_gen_shri_tl(cpu_tmp0, cpu_y, 1);

    tcg_gen_andi_tl(cpu_tmp0, cpu_tmp0, 0x7fffffff);

    tcg_gen_or_tl(cpu_tmp0, cpu_tmp0, r_temp);

    tcg_gen_andi_tl(cpu_y, cpu_tmp0, 0xffffffff);



    // b1 = N ^ V;

    gen_mov_reg_N(cpu_tmp0, cpu_psr);

    gen_mov_reg_V(r_temp, cpu_psr);

    tcg_gen_xor_tl(cpu_tmp0, cpu_tmp0, r_temp);

    tcg_temp_free(r_temp);



    // T0 = (b1 << 31) | (T0 >> 1);

    // src1 = T0;

    tcg_gen_shli_tl(cpu_tmp0, cpu_tmp0, 31);

    tcg_gen_shri_tl(cpu_cc_src, cpu_cc_src, 1);

    tcg_gen_or_tl(cpu_cc_src, cpu_cc_src, cpu_tmp0);



    tcg_gen_add_tl(cpu_cc_dst, cpu_cc_src, cpu_cc_src2);



    tcg_gen_mov_tl(dst, cpu_cc_dst);

}
