static void gen_pusha(DisasContext *s)

{

    int i;

    gen_op_movl_A0_reg(R_ESP);

    gen_op_addl_A0_im(-8 << s->dflag);

    if (!s->ss32)

        tcg_gen_ext16u_tl(cpu_A0, cpu_A0);

    tcg_gen_mov_tl(cpu_T[1], cpu_A0);

    if (s->addseg)

        gen_op_addl_A0_seg(s, R_SS);

    for(i = 0;i < 8; i++) {

        gen_op_mov_v_reg(MO_32, cpu_T[0], 7 - i);

        gen_op_st_v(s, s->dflag, cpu_T[0], cpu_A0);

        gen_op_addl_A0_im(1 << s->dflag);

    }

    gen_op_mov_reg_v(MO_16 + s->ss32, R_ESP, cpu_T[1]);

}
