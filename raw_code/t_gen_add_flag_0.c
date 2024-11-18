static inline void t_gen_add_flag(TCGv d, int flag)

{

	TCGv c;



	c = tcg_temp_new(TCG_TYPE_TL);

	t_gen_mov_TN_preg(c, PR_CCS);

	/* Propagate carry into d.  */

	tcg_gen_andi_tl(c, c, 1 << flag);

	if (flag)

		tcg_gen_shri_tl(c, c, flag);

	tcg_gen_add_tl(d, d, c);

	tcg_temp_free(c);

}
