static void gen_swa(DisasContext *dc, TCGv rb, TCGv ra, int32_t ofs)

{

    TCGv ea, val;

    TCGLabel *lab_fail, *lab_done;



    ea = tcg_temp_new();

    tcg_gen_addi_tl(ea, ra, ofs);



    lab_fail = gen_new_label();

    lab_done = gen_new_label();

    tcg_gen_brcond_tl(TCG_COND_NE, ea, cpu_lock_addr, lab_fail);

    tcg_temp_free(ea);



    val = tcg_temp_new();

    tcg_gen_atomic_cmpxchg_tl(val, cpu_lock_addr, cpu_lock_value,

                              rb, dc->mem_idx, MO_TEUL);

    tcg_gen_setcond_tl(TCG_COND_EQ, cpu_sr_f, val, cpu_lock_value);

    tcg_temp_free(val);



    tcg_gen_br(lab_done);



    gen_set_label(lab_fail);

    tcg_gen_movi_tl(cpu_sr_f, 0);



    gen_set_label(lab_done);

    tcg_gen_movi_tl(cpu_lock_addr, -1);

}
