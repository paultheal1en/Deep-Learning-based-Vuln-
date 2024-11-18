static void gen_farith (DisasContext *ctx, uint32_t op1, int ft,

                        int fs, int fd, int cc)

{

    const char *opn = "farith";

    const char *condnames[] = {

            "c.f",

            "c.un",

            "c.eq",

            "c.ueq",

            "c.olt",

            "c.ult",

            "c.ole",

            "c.ule",

            "c.sf",

            "c.ngle",

            "c.seq",

            "c.ngl",

            "c.lt",

            "c.nge",

            "c.le",

            "c.ngt",

    };

    const char *condnames_abs[] = {

            "cabs.f",

            "cabs.un",

            "cabs.eq",

            "cabs.ueq",

            "cabs.olt",

            "cabs.ult",

            "cabs.ole",

            "cabs.ule",

            "cabs.sf",

            "cabs.ngle",

            "cabs.seq",

            "cabs.ngl",

            "cabs.lt",

            "cabs.nge",

            "cabs.le",

            "cabs.ngt",

    };

    enum { BINOP, CMPOP, OTHEROP } optype = OTHEROP;

    uint32_t func = ctx->opcode & 0x3f;



    switch (ctx->opcode & FOP(0x3f, 0x1f)) {

    case FOP(0, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        gen_op_float_add_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "add.s";

        optype = BINOP;

        break;

    case FOP(1, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        gen_op_float_sub_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "sub.s";

        optype = BINOP;

        break;

    case FOP(2, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        gen_op_float_mul_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "mul.s";

        optype = BINOP;

        break;

    case FOP(3, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        gen_op_float_div_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "div.s";

        optype = BINOP;

        break;

    case FOP(4, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_sqrt_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "sqrt.s";

        break;

    case FOP(5, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_abs_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "abs.s";

        break;

    case FOP(6, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_mov_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "mov.s";

        break;

    case FOP(7, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_chs_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "neg.s";

        break;

    case FOP(8, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_roundl_s();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "round.l.s";

        break;

    case FOP(9, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_truncl_s();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "trunc.l.s";

        break;

    case FOP(10, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_ceill_s();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "ceil.l.s";

        break;

    case FOP(11, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_floorl_s();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "floor.l.s";

        break;

    case FOP(12, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_roundw_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "round.w.s";

        break;

    case FOP(13, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_truncw_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "trunc.w.s";

        break;

    case FOP(14, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_ceilw_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "ceil.w.s";

        break;

    case FOP(15, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_floorw_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "floor.w.s";

        break;

    case FOP(17, 16):

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        gen_movcf_s(ctx, (ft >> 2) & 0x7, ft & 0x1);

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "movcf.s";

        break;

    case FOP(18, 16):

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        gen_op_float_movz_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "movz.s";

        break;

    case FOP(19, 16):

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        gen_op_float_movn_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "movn.s";

        break;

    case FOP(21, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_recip_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "recip.s";

        break;

    case FOP(22, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_rsqrt_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "rsqrt.s";

        break;

    case FOP(28, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        gen_op_float_recip2_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "recip2.s";

        break;

    case FOP(29, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_recip1_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "recip1.s";

        break;

    case FOP(30, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_rsqrt1_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "rsqrt1.s";

        break;

    case FOP(31, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        gen_op_float_rsqrt2_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "rsqrt2.s";

        break;

    case FOP(33, 16):

        gen_op_cp1_registers(fd);

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_cvtd_s();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "cvt.d.s";

        break;

    case FOP(36, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_cvtw_s();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "cvt.w.s";

        break;

    case FOP(37, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_cvtl_s();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "cvt.l.s";

        break;

    case FOP(38, 16):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT1, fs);

        GEN_LOAD_FREG_FTN(WT0, ft);

        gen_op_float_cvtps_s();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "cvt.ps.s";

        break;

    case FOP(48, 16):

    case FOP(49, 16):

    case FOP(50, 16):

    case FOP(51, 16):

    case FOP(52, 16):

    case FOP(53, 16):

    case FOP(54, 16):

    case FOP(55, 16):

    case FOP(56, 16):

    case FOP(57, 16):

    case FOP(58, 16):

    case FOP(59, 16):

    case FOP(60, 16):

    case FOP(61, 16):

    case FOP(62, 16):

    case FOP(63, 16):

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        if (ctx->opcode & (1 << 6)) {

            gen_op_cp1_64bitmode();

            gen_cmpabs_s(func-48, cc);

            opn = condnames_abs[func-48];

        } else {

            gen_cmp_s(func-48, cc);

            opn = condnames[func-48];

        }

        break;

    case FOP(0, 17):

        gen_op_cp1_registers(fs | ft | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        gen_op_float_add_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "add.d";

        optype = BINOP;

        break;

    case FOP(1, 17):

        gen_op_cp1_registers(fs | ft | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        gen_op_float_sub_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "sub.d";

        optype = BINOP;

        break;

    case FOP(2, 17):

        gen_op_cp1_registers(fs | ft | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        gen_op_float_mul_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "mul.d";

        optype = BINOP;

        break;

    case FOP(3, 17):

        gen_op_cp1_registers(fs | ft | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        gen_op_float_div_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "div.d";

        optype = BINOP;

        break;

    case FOP(4, 17):

        gen_op_cp1_registers(fs | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_sqrt_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "sqrt.d";

        break;

    case FOP(5, 17):

        gen_op_cp1_registers(fs | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_abs_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "abs.d";

        break;

    case FOP(6, 17):

        gen_op_cp1_registers(fs | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_mov_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "mov.d";

        break;

    case FOP(7, 17):

        gen_op_cp1_registers(fs | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_chs_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "neg.d";

        break;

    case FOP(8, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_roundl_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "round.l.d";

        break;

    case FOP(9, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_truncl_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "trunc.l.d";

        break;

    case FOP(10, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_ceill_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "ceil.l.d";

        break;

    case FOP(11, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_floorl_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "floor.l.d";

        break;

    case FOP(12, 17):

        gen_op_cp1_registers(fs);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_roundw_d();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "round.w.d";

        break;

    case FOP(13, 17):

        gen_op_cp1_registers(fs);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_truncw_d();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "trunc.w.d";

        break;

    case FOP(14, 17):

        gen_op_cp1_registers(fs);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_ceilw_d();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "ceil.w.d";

        break;

    case FOP(15, 17):

        gen_op_cp1_registers(fs);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_floorw_d();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "floor.w.d";

        break;

    case FOP(17, 17):

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT2, fd);

        gen_movcf_d(ctx, (ft >> 2) & 0x7, ft & 0x1);

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "movcf.d";

        break;

    case FOP(18, 17):

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT2, fd);

        gen_op_float_movz_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "movz.d";

        break;

    case FOP(19, 17):

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT2, fd);

        gen_op_float_movn_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "movn.d";

        break;

    case FOP(21, 17):

        gen_op_cp1_registers(fs | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_recip_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "recip.d";

        break;

    case FOP(22, 17):

        gen_op_cp1_registers(fs | fd);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_rsqrt_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "rsqrt.d";

        break;

    case FOP(28, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT2, ft);

        gen_op_float_recip2_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "recip2.d";

        break;

    case FOP(29, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_recip1_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "recip1.d";

        break;

    case FOP(30, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_rsqrt1_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "rsqrt1.d";

        break;

    case FOP(31, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT2, ft);

        gen_op_float_rsqrt2_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "rsqrt2.d";

        break;

    case FOP(48, 17):

    case FOP(49, 17):

    case FOP(50, 17):

    case FOP(51, 17):

    case FOP(52, 17):

    case FOP(53, 17):

    case FOP(54, 17):

    case FOP(55, 17):

    case FOP(56, 17):

    case FOP(57, 17):

    case FOP(58, 17):

    case FOP(59, 17):

    case FOP(60, 17):

    case FOP(61, 17):

    case FOP(62, 17):

    case FOP(63, 17):

        GEN_LOAD_FREG_FTN(DT0, fs);

        GEN_LOAD_FREG_FTN(DT1, ft);

        if (ctx->opcode & (1 << 6)) {

            gen_op_cp1_64bitmode();

            gen_cmpabs_d(func-48, cc);

            opn = condnames_abs[func-48];

        } else {

            gen_op_cp1_registers(fs | ft);

            gen_cmp_d(func-48, cc);

            opn = condnames[func-48];

        }

        break;

    case FOP(32, 17):

        gen_op_cp1_registers(fs);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_cvts_d();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "cvt.s.d";

        break;

    case FOP(36, 17):

        gen_op_cp1_registers(fs);

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_cvtw_d();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "cvt.w.d";

        break;

    case FOP(37, 17):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_cvtl_d();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "cvt.l.d";

        break;

    case FOP(32, 20):

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_cvts_w();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "cvt.s.w";

        break;

    case FOP(33, 20):

        gen_op_cp1_registers(fd);

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_cvtd_w();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "cvt.d.w";

        break;

    case FOP(32, 21):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_cvts_l();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "cvt.s.l";

        break;

    case FOP(33, 21):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(DT0, fs);

        gen_op_float_cvtd_l();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "cvt.d.l";

        break;

    case FOP(38, 20):

    case FOP(38, 21):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_cvtps_pw();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "cvt.ps.pw";

        break;

    case FOP(0, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        gen_op_float_add_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "add.ps";

        break;

    case FOP(1, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        gen_op_float_sub_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "sub.ps";

        break;

    case FOP(2, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        gen_op_float_mul_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "mul.ps";

        break;

    case FOP(5, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_abs_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "abs.ps";

        break;

    case FOP(6, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_mov_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "mov.ps";

        break;

    case FOP(7, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_chs_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "neg.ps";

        break;

    case FOP(17, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        GEN_LOAD_FREG_FTN(WTH2, fd);

        gen_movcf_ps(ctx, (ft >> 2) & 0x7, ft & 0x1);

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "movcf.ps";

        break;

    case FOP(18, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        GEN_LOAD_FREG_FTN(WTH2, fd);

        gen_op_float_movz_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "movz.ps";

        break;

    case FOP(19, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_REG_TN(T0, ft);

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        GEN_LOAD_FREG_FTN(WTH2, fd);

        gen_op_float_movn_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "movn.ps";

        break;

    case FOP(24, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, ft);

        GEN_LOAD_FREG_FTN(WTH0, ft);

        GEN_LOAD_FREG_FTN(WT1, fs);

        GEN_LOAD_FREG_FTN(WTH1, fs);

        gen_op_float_addr_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "addr.ps";

        break;

    case FOP(26, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, ft);

        GEN_LOAD_FREG_FTN(WTH0, ft);

        GEN_LOAD_FREG_FTN(WT1, fs);

        GEN_LOAD_FREG_FTN(WTH1, fs);

        gen_op_float_mulr_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "mulr.ps";

        break;

    case FOP(28, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        GEN_LOAD_FREG_FTN(WTH2, fd);

        gen_op_float_recip2_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "recip2.ps";

        break;

    case FOP(29, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_recip1_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "recip1.ps";

        break;

    case FOP(30, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_rsqrt1_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "rsqrt1.ps";

        break;

    case FOP(31, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT2, fd);

        GEN_LOAD_FREG_FTN(WTH2, fd);

        gen_op_float_rsqrt2_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "rsqrt2.ps";

        break;

    case FOP(32, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_cvts_pu();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "cvt.s.pu";

        break;

    case FOP(36, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        gen_op_float_cvtpw_ps();

        GEN_STORE_FTN_FREG(fd, WT2);

        GEN_STORE_FTN_FREG(fd, WTH2);

        opn = "cvt.pw.ps";

        break;

    case FOP(40, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        gen_op_float_cvts_pl();

        GEN_STORE_FTN_FREG(fd, WT2);

        opn = "cvt.s.pl";

        break;

    case FOP(44, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        gen_op_float_pll_ps();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "pll.ps";

        break;

    case FOP(45, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        gen_op_float_plu_ps();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "plu.ps";

        break;

    case FOP(46, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        gen_op_float_pul_ps();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "pul.ps";

        break;

    case FOP(47, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        gen_op_float_puu_ps();

        GEN_STORE_FTN_FREG(fd, DT2);

        opn = "puu.ps";

        break;

    case FOP(48, 22):

    case FOP(49, 22):

    case FOP(50, 22):

    case FOP(51, 22):

    case FOP(52, 22):

    case FOP(53, 22):

    case FOP(54, 22):

    case FOP(55, 22):

    case FOP(56, 22):

    case FOP(57, 22):

    case FOP(58, 22):

    case FOP(59, 22):

    case FOP(60, 22):

    case FOP(61, 22):

    case FOP(62, 22):

    case FOP(63, 22):

        gen_op_cp1_64bitmode();

        GEN_LOAD_FREG_FTN(WT0, fs);

        GEN_LOAD_FREG_FTN(WTH0, fs);

        GEN_LOAD_FREG_FTN(WT1, ft);

        GEN_LOAD_FREG_FTN(WTH1, ft);

        if (ctx->opcode & (1 << 6)) {

            gen_cmpabs_ps(func-48, cc);

            opn = condnames_abs[func-48];

        } else {

            gen_cmp_ps(func-48, cc);

            opn = condnames[func-48];

        }

        break;

    default:

        MIPS_INVAL(opn);

        generate_exception (ctx, EXCP_RI);

        return;

    }

    switch (optype) {

    case BINOP:

        MIPS_DEBUG("%s %s, %s, %s", opn, fregnames[fd], fregnames[fs], fregnames[ft]);

        break;

    case CMPOP:

        MIPS_DEBUG("%s %s,%s", opn, fregnames[fs], fregnames[ft]);

        break;

    default:

        MIPS_DEBUG("%s %s,%s", opn, fregnames[fd], fregnames[fs]);

        break;

    }

}
