static unsigned int dec_movu_r(DisasContext *dc)

{

	TCGv t0;

	int size = memsize_z(dc);

	DIS(fprintf (logfile, "movu.%c $r%u, $r%u\n",

		    memsize_char(size),

		    dc->op1, dc->op2));



	cris_cc_mask(dc, CC_MASK_NZ);

	t0 = tcg_temp_new(TCG_TYPE_TL);

	dec_prep_move_r(dc, dc->op1, dc->op2, size, 0, t0);

	cris_alu(dc, CC_OP_MOVE, cpu_R[dc->op2], cpu_R[dc->op2], t0, 4);

	tcg_temp_free(t0);

	return 2;

}
