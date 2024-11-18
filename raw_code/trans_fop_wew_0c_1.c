static ExitStatus trans_fop_wew_0c(DisasContext *ctx, uint32_t insn,

                                   const DisasInsn *di)

{

    unsigned rt = extract32(insn, 0, 5);

    unsigned ra = extract32(insn, 21, 5);

    return do_fop_wew(ctx, rt, ra, di->f_wew);

}
