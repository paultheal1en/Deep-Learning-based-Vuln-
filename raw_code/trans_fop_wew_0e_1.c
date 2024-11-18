static ExitStatus trans_fop_wew_0e(DisasContext *ctx, uint32_t insn,

                                   const DisasInsn *di)

{

    unsigned rt = assemble_rt64(insn);

    unsigned ra = assemble_ra64(insn);

    return do_fop_wew(ctx, rt, ra, di->f_wew);

}
