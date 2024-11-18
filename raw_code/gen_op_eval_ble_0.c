static inline void gen_op_eval_ble(TCGv dst, TCGv_i32 src)

{

    gen_mov_reg_N(cpu_tmp0, src);

    gen_mov_reg_V(dst, src);

    tcg_gen_xor_tl(dst, dst, cpu_tmp0);

    gen_mov_reg_Z(cpu_tmp0, src);

    tcg_gen_or_tl(dst, dst, cpu_tmp0);

}
