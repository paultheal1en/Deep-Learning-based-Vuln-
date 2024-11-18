static ExitStatus trans_fop_weww_0e(DisasContext *ctx, uint32_t insn,

                                    const DisasInsn *di)

{

    unsigned rt = assemble_rt64(insn);

    unsigned rb = assemble_rb64(insn);

    unsigned ra = assemble_ra64(insn);

    return do_fop_weww(ctx, rt, ra, rb, di->f_weww);

}
