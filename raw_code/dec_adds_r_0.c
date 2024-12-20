static unsigned int dec_adds_r(DisasContext *dc)

{

	TCGv t0;

	int size = memsize_z(dc);

	DIS(fprintf (logfile, "adds.%c $r%u, $r%u\n",

		    memsize_char(size),

		    dc->op1, dc->op2));



	cris_cc_mask(dc, CC_MASK_NZVC);

	t0 = tcg_temp_new(TCG_TYPE_TL);

	/* Size can only be qi or hi.  */

	t_gen_sext(t0, cpu_R[dc->op1], size);

	cris_alu(dc, CC_OP_ADD,

		    cpu_R[dc->op2], cpu_R[dc->op2], t0, 4);

	tcg_temp_free(t0);

	return 2;

}
