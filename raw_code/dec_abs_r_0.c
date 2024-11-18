static unsigned int dec_abs_r(DisasContext *dc)

{

	TCGv t0;



	DIS(fprintf (logfile, "abs $r%u, $r%u\n",

		    dc->op1, dc->op2));

	cris_cc_mask(dc, CC_MASK_NZ);



	t0 = tcg_temp_new(TCG_TYPE_TL);

	tcg_gen_sari_tl(t0, cpu_R[dc->op1], 31);

	tcg_gen_xor_tl(cpu_R[dc->op2], cpu_R[dc->op1], t0);

	tcg_gen_sub_tl(cpu_R[dc->op2], cpu_R[dc->op2], t0);

	tcg_temp_free(t0);



	cris_alu(dc, CC_OP_MOVE,

		    cpu_R[dc->op2], cpu_R[dc->op2], cpu_R[dc->op2], 4);

	return 2;

}
