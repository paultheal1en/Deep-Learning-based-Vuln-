static inline void t_gen_subx_carry(DisasContext *dc, TCGv d)

{

	if (dc->flagx_known) {

		if (dc->flags_x) {

			TCGv c;

            

			c = tcg_temp_new(TCG_TYPE_TL);

			t_gen_mov_TN_preg(c, PR_CCS);

			/* C flag is already at bit 0.  */

			tcg_gen_andi_tl(c, c, C_FLAG);

			tcg_gen_sub_tl(d, d, c);

			tcg_temp_free(c);

		}

	} else {

		TCGv x, c;



		x = tcg_temp_new(TCG_TYPE_TL);

		c = tcg_temp_new(TCG_TYPE_TL);

		t_gen_mov_TN_preg(x, PR_CCS);

		tcg_gen_mov_tl(c, x);



		/* Propagate carry into d if X is set. Branch free.  */

		tcg_gen_andi_tl(c, c, C_FLAG);

		tcg_gen_andi_tl(x, x, X_FLAG);

		tcg_gen_shri_tl(x, x, 4);



		tcg_gen_and_tl(x, x, c);

		tcg_gen_sub_tl(d, d, x);

		tcg_temp_free(x);

		tcg_temp_free(c);

	}

}
