static unsigned int dec_move_rs(DisasContext *dc)

{

	DIS(fprintf (logfile, "move $r%u, $s%u\n", dc->op1, dc->op2));

	cris_cc_mask(dc, 0);

	tcg_gen_helper_0_2(helper_movl_sreg_reg, 

			   tcg_const_tl(dc->op2), tcg_const_tl(dc->op1));

	return 2;

}
