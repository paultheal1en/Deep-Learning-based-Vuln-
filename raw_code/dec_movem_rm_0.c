static unsigned int dec_movem_rm(DisasContext *dc)

{

	TCGv tmp;

	TCGv addr;

	int i;



	DIS(fprintf (logfile, "movem $r%u, [$r%u%s\n", dc->op2, dc->op1,

		     dc->postinc ? "+]" : "]"));



	cris_flush_cc_state(dc);



	tmp = tcg_temp_new(TCG_TYPE_TL);

	addr = tcg_temp_new(TCG_TYPE_TL);

	tcg_gen_movi_tl(tmp, 4);

	tcg_gen_mov_tl(addr, cpu_R[dc->op1]);

	for (i = 0; i <= dc->op2; i++) {

		/* Displace addr.  */

		/* Perform the store.  */

		gen_store(dc, addr, cpu_R[i], 4);

		tcg_gen_add_tl(addr, addr, tmp);

	}

	if (dc->postinc)

		tcg_gen_mov_tl(cpu_R[dc->op1], addr);

	cris_cc_mask(dc, 0);

	tcg_temp_free(tmp);

	tcg_temp_free(addr);

	return 2;

}
