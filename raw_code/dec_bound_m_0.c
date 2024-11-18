static unsigned int dec_bound_m(DisasContext *dc)

{

	TCGv l[2];

	int memsize = memsize_zz(dc);

	int insn_len;

	DIS(fprintf (logfile, "bound.%d [$r%u%s, $r%u\n",

		    memsize_char(memsize),

		    dc->op1, dc->postinc ? "+]" : "]",

		    dc->op2));



	l[0] = tcg_temp_local_new(TCG_TYPE_TL);

	l[1] = tcg_temp_local_new(TCG_TYPE_TL);

	insn_len = dec_prep_alu_m(dc, 0, memsize, l[0], l[1]);

	cris_cc_mask(dc, CC_MASK_NZ);

	cris_alu(dc, CC_OP_BOUND, cpu_R[dc->op2], l[0], l[1], 4);

	do_postinc(dc, memsize);

	tcg_temp_free(l[0]);

	tcg_temp_free(l[1]);

	return insn_len;

}
