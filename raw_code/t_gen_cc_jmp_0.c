static void t_gen_cc_jmp(TCGv pc_true, TCGv pc_false)

{

	TCGv btaken;

	int l1;



	l1 = gen_new_label();

	btaken = tcg_temp_new(TCG_TYPE_TL);



	/* Conditional jmp.  */

	tcg_gen_mov_tl(btaken, env_btaken);

	tcg_gen_mov_tl(env_pc, pc_false);

	tcg_gen_brcondi_tl(TCG_COND_EQ, btaken, 0, l1);

	tcg_gen_mov_tl(env_pc, pc_true);

	gen_set_label(l1);



	tcg_temp_free(btaken);

}
