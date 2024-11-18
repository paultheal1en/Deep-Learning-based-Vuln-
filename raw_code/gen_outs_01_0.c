static inline void gen_outs(DisasContext *s, int ot)

{

    gen_string_movl_A0_ESI(s);

    gen_op_ld_T0_A0(ot + s->mem_index);



    gen_op_mov_TN_reg(OT_WORD, 1, R_EDX);

    tcg_gen_trunc_tl_i32(cpu_tmp2_i32, cpu_T[1]);

    tcg_gen_andi_i32(cpu_tmp2_i32, cpu_tmp2_i32, 0xffff);

    tcg_gen_trunc_tl_i32(cpu_tmp3_i32, cpu_T[0]);

    tcg_gen_helper_0_2(helper_out_func[ot], cpu_tmp2_i32, cpu_tmp3_i32);



    gen_op_movl_T0_Dshift[ot]();

#ifdef TARGET_X86_64

    if (s->aflag == 2) {

        gen_op_addq_ESI_T0();

    } else

#endif

    if (s->aflag) {

        gen_op_addl_ESI_T0();

    } else {

        gen_op_addw_ESI_T0();

    }

}
