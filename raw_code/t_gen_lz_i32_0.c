static void t_gen_lz_i32(TCGv d, TCGv x)

{

	TCGv y, m, n;



	y = tcg_temp_new(TCG_TYPE_I32);

	m = tcg_temp_new(TCG_TYPE_I32);

	n = tcg_temp_new(TCG_TYPE_I32);



	/* y = -(x >> 16)  */

	tcg_gen_shri_i32(y, x, 16);

	tcg_gen_neg_i32(y, y);



	/* m = (y >> 16) & 16  */

	tcg_gen_sari_i32(m, y, 16);

	tcg_gen_andi_i32(m, m, 16);



	/* n = 16 - m  */

	tcg_gen_sub_i32(n, tcg_const_i32(16), m);

	/* x = x >> m  */

	tcg_gen_shr_i32(x, x, m);



	/* y = x - 0x100  */

	tcg_gen_subi_i32(y, x, 0x100);

	/* m = (y >> 16) & 8  */

	tcg_gen_sari_i32(m, y, 16);

	tcg_gen_andi_i32(m, m, 8);

	/* n = n + m  */

	tcg_gen_add_i32(n, n, m);

	/* x = x << m  */

	tcg_gen_shl_i32(x, x, m);



	/* y = x - 0x1000  */

	tcg_gen_subi_i32(y, x, 0x1000);

	/* m = (y >> 16) & 4  */

	tcg_gen_sari_i32(m, y, 16);

	tcg_gen_andi_i32(m, m, 4);

	/* n = n + m  */

	tcg_gen_add_i32(n, n, m);

	/* x = x << m  */

	tcg_gen_shl_i32(x, x, m);



	/* y = x - 0x4000  */

	tcg_gen_subi_i32(y, x, 0x4000);

	/* m = (y >> 16) & 2  */

	tcg_gen_sari_i32(m, y, 16);

	tcg_gen_andi_i32(m, m, 2);

	/* n = n + m  */

	tcg_gen_add_i32(n, n, m);

	/* x = x << m  */

	tcg_gen_shl_i32(x, x, m);



	/* y = x >> 14  */

	tcg_gen_shri_i32(y, x, 14);

	/* m = y & ~(y >> 1)  */

	tcg_gen_sari_i32(m, y, 1);

	tcg_gen_not_i32(m, m);

	tcg_gen_and_i32(m, m, y);



	/* d = n + 2 - m  */

	tcg_gen_addi_i32(d, n, 2);

	tcg_gen_sub_i32(d, d, m);



	tcg_temp_free(y);

	tcg_temp_free(m);

	tcg_temp_free(n);

}
