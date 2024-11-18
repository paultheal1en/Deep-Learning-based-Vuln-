static inline void gen_op_eval_bleu(TCGv dst, TCGv_i32 src)

{

    gen_mov_reg_Z(cpu_tmp0, src);

    gen_mov_reg_C(dst, src);

    tcg_gen_or_tl(dst, dst, cpu_tmp0);

}
