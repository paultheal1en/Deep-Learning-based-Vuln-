static inline void t_gen_swapb(TCGv d, TCGv s)

{

	TCGv t, org_s;



	t = tcg_temp_new(TCG_TYPE_TL);

	org_s = tcg_temp_new(TCG_TYPE_TL);



	/* d and s may refer to the same object.  */

	tcg_gen_mov_tl(org_s, s);

	tcg_gen_shli_tl(t, org_s, 8);

	tcg_gen_andi_tl(d, t, 0xff00ff00);

	tcg_gen_shri_tl(t, org_s, 8);

	tcg_gen_andi_tl(t, t, 0x00ff00ff);

	tcg_gen_or_tl(d, d, t);

	tcg_temp_free(t);

	tcg_temp_free(org_s);

}
