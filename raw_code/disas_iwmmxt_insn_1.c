static int disas_iwmmxt_insn(CPUState *env, DisasContext *s, uint32_t insn)

{

    int rd, wrd;

    int rdhi, rdlo, rd0, rd1, i;

    TCGv addr;

    TCGv tmp, tmp2, tmp3;



    if ((insn & 0x0e000e00) == 0x0c000000) {

        if ((insn & 0x0fe00ff0) == 0x0c400000) {

            wrd = insn & 0xf;

            rdlo = (insn >> 12) & 0xf;

            rdhi = (insn >> 16) & 0xf;

            if (insn & ARM_CP_RW_BIT) {			/* TMRRC */

                iwmmxt_load_reg(cpu_V0, wrd);

                tcg_gen_trunc_i64_i32(cpu_R[rdlo], cpu_V0);

                tcg_gen_shri_i64(cpu_V0, cpu_V0, 32);

                tcg_gen_trunc_i64_i32(cpu_R[rdhi], cpu_V0);

            } else {					/* TMCRR */

                tcg_gen_concat_i32_i64(cpu_V0, cpu_R[rdlo], cpu_R[rdhi]);

                iwmmxt_store_reg(cpu_V0, wrd);

                gen_op_iwmmxt_set_mup();

            }

            return 0;

        }



        wrd = (insn >> 12) & 0xf;

        addr = new_tmp();

        if (gen_iwmmxt_address(s, insn, addr)) {


            return 1;

        }

        if (insn & ARM_CP_RW_BIT) {

            if ((insn >> 28) == 0xf) {			/* WLDRW wCx */

                tmp = new_tmp();

                tcg_gen_qemu_ld32u(tmp, addr, IS_USER(s));

                iwmmxt_store_creg(wrd, tmp);

            } else {

                i = 1;

                if (insn & (1 << 8)) {

                    if (insn & (1 << 22)) {		/* WLDRD */

                        tcg_gen_qemu_ld64(cpu_M0, addr, IS_USER(s));

                        i = 0;

                    } else {				/* WLDRW wRd */

                        tmp = gen_ld32(addr, IS_USER(s));

                    }

                } else {

                    if (insn & (1 << 22)) {		/* WLDRH */

                        tmp = gen_ld16u(addr, IS_USER(s));

                    } else {				/* WLDRB */

                        tmp = gen_ld8u(addr, IS_USER(s));

                    }

                }

                if (i) {

                    tcg_gen_extu_i32_i64(cpu_M0, tmp);

                    dead_tmp(tmp);

                }

                gen_op_iwmmxt_movq_wRn_M0(wrd);

            }

        } else {

            if ((insn >> 28) == 0xf) {			/* WSTRW wCx */

                tmp = iwmmxt_load_creg(wrd);

                gen_st32(tmp, addr, IS_USER(s));

            } else {

                gen_op_iwmmxt_movq_M0_wRn(wrd);

                tmp = new_tmp();

                if (insn & (1 << 8)) {

                    if (insn & (1 << 22)) {		/* WSTRD */

                        dead_tmp(tmp);

                        tcg_gen_qemu_st64(cpu_M0, addr, IS_USER(s));

                    } else {				/* WSTRW wRd */

                        tcg_gen_trunc_i64_i32(tmp, cpu_M0);

                        gen_st32(tmp, addr, IS_USER(s));

                    }

                } else {

                    if (insn & (1 << 22)) {		/* WSTRH */

                        tcg_gen_trunc_i64_i32(tmp, cpu_M0);

                        gen_st16(tmp, addr, IS_USER(s));

                    } else {				/* WSTRB */

                        tcg_gen_trunc_i64_i32(tmp, cpu_M0);

                        gen_st8(tmp, addr, IS_USER(s));

                    }

                }

            }

        }


        return 0;

    }



    if ((insn & 0x0f000000) != 0x0e000000)

        return 1;



    switch (((insn >> 12) & 0xf00) | ((insn >> 4) & 0xff)) {

    case 0x000:						/* WOR */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 0) & 0xf;

        rd1 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        gen_op_iwmmxt_orq_M0_wRn(rd1);

        gen_op_iwmmxt_setpsr_nz();

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x011:						/* TMCR */

        if (insn & 0xf)

            return 1;

        rd = (insn >> 12) & 0xf;

        wrd = (insn >> 16) & 0xf;

        switch (wrd) {

        case ARM_IWMMXT_wCID:

        case ARM_IWMMXT_wCASF:

            break;

        case ARM_IWMMXT_wCon:

            gen_op_iwmmxt_set_cup();

            /* Fall through.  */

        case ARM_IWMMXT_wCSSF:

            tmp = iwmmxt_load_creg(wrd);

            tmp2 = load_reg(s, rd);

            tcg_gen_andc_i32(tmp, tmp, tmp2);

            dead_tmp(tmp2);

            iwmmxt_store_creg(wrd, tmp);

            break;

        case ARM_IWMMXT_wCGR0:

        case ARM_IWMMXT_wCGR1:

        case ARM_IWMMXT_wCGR2:

        case ARM_IWMMXT_wCGR3:

            gen_op_iwmmxt_set_cup();

            tmp = load_reg(s, rd);

            iwmmxt_store_creg(wrd, tmp);

            break;

        default:

            return 1;

        }

        break;

    case 0x100:						/* WXOR */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 0) & 0xf;

        rd1 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        gen_op_iwmmxt_xorq_M0_wRn(rd1);

        gen_op_iwmmxt_setpsr_nz();

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x111:						/* TMRC */

        if (insn & 0xf)

            return 1;

        rd = (insn >> 12) & 0xf;

        wrd = (insn >> 16) & 0xf;

        tmp = iwmmxt_load_creg(wrd);

        store_reg(s, rd, tmp);

        break;

    case 0x300:						/* WANDN */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 0) & 0xf;

        rd1 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tcg_gen_neg_i64(cpu_M0, cpu_M0);

        gen_op_iwmmxt_andq_M0_wRn(rd1);

        gen_op_iwmmxt_setpsr_nz();

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x200:						/* WAND */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 0) & 0xf;

        rd1 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        gen_op_iwmmxt_andq_M0_wRn(rd1);

        gen_op_iwmmxt_setpsr_nz();

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x810: case 0xa10:				/* WMADD */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 0) & 0xf;

        rd1 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        if (insn & (1 << 21))

            gen_op_iwmmxt_maddsq_M0_wRn(rd1);

        else

            gen_op_iwmmxt_madduq_M0_wRn(rd1);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x10e: case 0x50e: case 0x90e: case 0xd0e:	/* WUNPCKIL */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            gen_op_iwmmxt_unpacklb_M0_wRn(rd1);

            break;

        case 1:

            gen_op_iwmmxt_unpacklw_M0_wRn(rd1);

            break;

        case 2:

            gen_op_iwmmxt_unpackll_M0_wRn(rd1);

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x10c: case 0x50c: case 0x90c: case 0xd0c:	/* WUNPCKIH */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            gen_op_iwmmxt_unpackhb_M0_wRn(rd1);

            break;

        case 1:

            gen_op_iwmmxt_unpackhw_M0_wRn(rd1);

            break;

        case 2:

            gen_op_iwmmxt_unpackhl_M0_wRn(rd1);

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x012: case 0x112: case 0x412: case 0x512:	/* WSAD */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        if (insn & (1 << 22))

            gen_op_iwmmxt_sadw_M0_wRn(rd1);

        else

            gen_op_iwmmxt_sadb_M0_wRn(rd1);

        if (!(insn & (1 << 20)))

            gen_op_iwmmxt_addl_M0_wRn(wrd);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x010: case 0x110: case 0x210: case 0x310:	/* WMUL */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        if (insn & (1 << 21)) {

            if (insn & (1 << 20))

                gen_op_iwmmxt_mulshw_M0_wRn(rd1);

            else

                gen_op_iwmmxt_mulslw_M0_wRn(rd1);

        } else {

            if (insn & (1 << 20))

                gen_op_iwmmxt_muluhw_M0_wRn(rd1);

            else

                gen_op_iwmmxt_mululw_M0_wRn(rd1);

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x410: case 0x510: case 0x610: case 0x710:	/* WMAC */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        if (insn & (1 << 21))

            gen_op_iwmmxt_macsw_M0_wRn(rd1);

        else

            gen_op_iwmmxt_macuw_M0_wRn(rd1);

        if (!(insn & (1 << 20))) {

            iwmmxt_load_reg(cpu_V1, wrd);

            tcg_gen_add_i64(cpu_M0, cpu_M0, cpu_V1);

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x006: case 0x406: case 0x806: case 0xc06:	/* WCMPEQ */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            gen_op_iwmmxt_cmpeqb_M0_wRn(rd1);

            break;

        case 1:

            gen_op_iwmmxt_cmpeqw_M0_wRn(rd1);

            break;

        case 2:

            gen_op_iwmmxt_cmpeql_M0_wRn(rd1);

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x800: case 0x900: case 0xc00: case 0xd00:	/* WAVG2 */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        if (insn & (1 << 22)) {

            if (insn & (1 << 20))

                gen_op_iwmmxt_avgw1_M0_wRn(rd1);

            else

                gen_op_iwmmxt_avgw0_M0_wRn(rd1);

        } else {

            if (insn & (1 << 20))

                gen_op_iwmmxt_avgb1_M0_wRn(rd1);

            else

                gen_op_iwmmxt_avgb0_M0_wRn(rd1);

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x802: case 0x902: case 0xa02: case 0xb02:	/* WALIGNR */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = iwmmxt_load_creg(ARM_IWMMXT_wCGR0 + ((insn >> 20) & 3));

        tcg_gen_andi_i32(tmp, tmp, 7);

        iwmmxt_load_reg(cpu_V1, rd1);

        gen_helper_iwmmxt_align(cpu_M0, cpu_M0, cpu_V1, tmp);

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x601: case 0x605: case 0x609: case 0x60d:	/* TINSR */

        if (((insn >> 6) & 3) == 3)

            return 1;

        rd = (insn >> 12) & 0xf;

        wrd = (insn >> 16) & 0xf;

        tmp = load_reg(s, rd);

        gen_op_iwmmxt_movq_M0_wRn(wrd);

        switch ((insn >> 6) & 3) {

        case 0:

            tmp2 = tcg_const_i32(0xff);

            tmp3 = tcg_const_i32((insn & 7) << 3);

            break;

        case 1:

            tmp2 = tcg_const_i32(0xffff);

            tmp3 = tcg_const_i32((insn & 3) << 4);

            break;

        case 2:

            tmp2 = tcg_const_i32(0xffffffff);

            tmp3 = tcg_const_i32((insn & 1) << 5);

            break;

        default:

            TCGV_UNUSED(tmp2);

            TCGV_UNUSED(tmp3);

        }

        gen_helper_iwmmxt_insr(cpu_M0, cpu_M0, tmp, tmp2, tmp3);

        tcg_temp_free(tmp3);

        tcg_temp_free(tmp2);

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x107: case 0x507: case 0x907: case 0xd07:	/* TEXTRM */

        rd = (insn >> 12) & 0xf;

        wrd = (insn >> 16) & 0xf;

        if (rd == 15 || ((insn >> 22) & 3) == 3)

            return 1;

        gen_op_iwmmxt_movq_M0_wRn(wrd);

        tmp = new_tmp();

        switch ((insn >> 22) & 3) {

        case 0:

            tcg_gen_shri_i64(cpu_M0, cpu_M0, (insn & 7) << 3);

            tcg_gen_trunc_i64_i32(tmp, cpu_M0);

            if (insn & 8) {

                tcg_gen_ext8s_i32(tmp, tmp);

            } else {

                tcg_gen_andi_i32(tmp, tmp, 0xff);

            }

            break;

        case 1:

            tcg_gen_shri_i64(cpu_M0, cpu_M0, (insn & 3) << 4);

            tcg_gen_trunc_i64_i32(tmp, cpu_M0);

            if (insn & 8) {

                tcg_gen_ext16s_i32(tmp, tmp);

            } else {

                tcg_gen_andi_i32(tmp, tmp, 0xffff);

            }

            break;

        case 2:

            tcg_gen_shri_i64(cpu_M0, cpu_M0, (insn & 1) << 5);

            tcg_gen_trunc_i64_i32(tmp, cpu_M0);

            break;

        }

        store_reg(s, rd, tmp);

        break;

    case 0x117: case 0x517: case 0x917: case 0xd17:	/* TEXTRC */

        if ((insn & 0x000ff008) != 0x0003f000 || ((insn >> 22) & 3) == 3)

            return 1;

        tmp = iwmmxt_load_creg(ARM_IWMMXT_wCASF);

        switch ((insn >> 22) & 3) {

        case 0:

            tcg_gen_shri_i32(tmp, tmp, ((insn & 7) << 2) + 0);

            break;

        case 1:

            tcg_gen_shri_i32(tmp, tmp, ((insn & 3) << 3) + 4);

            break;

        case 2:

            tcg_gen_shri_i32(tmp, tmp, ((insn & 1) << 4) + 12);

            break;

        }

        tcg_gen_shli_i32(tmp, tmp, 28);

        gen_set_nzcv(tmp);

        dead_tmp(tmp);

        break;

    case 0x401: case 0x405: case 0x409: case 0x40d:	/* TBCST */

        if (((insn >> 6) & 3) == 3)

            return 1;

        rd = (insn >> 12) & 0xf;

        wrd = (insn >> 16) & 0xf;

        tmp = load_reg(s, rd);

        switch ((insn >> 6) & 3) {

        case 0:

            gen_helper_iwmmxt_bcstb(cpu_M0, tmp);

            break;

        case 1:

            gen_helper_iwmmxt_bcstw(cpu_M0, tmp);

            break;

        case 2:

            gen_helper_iwmmxt_bcstl(cpu_M0, tmp);

            break;

        }

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x113: case 0x513: case 0x913: case 0xd13:	/* TANDC */

        if ((insn & 0x000ff00f) != 0x0003f000 || ((insn >> 22) & 3) == 3)

            return 1;

        tmp = iwmmxt_load_creg(ARM_IWMMXT_wCASF);

        tmp2 = new_tmp();

        tcg_gen_mov_i32(tmp2, tmp);

        switch ((insn >> 22) & 3) {

        case 0:

            for (i = 0; i < 7; i ++) {

                tcg_gen_shli_i32(tmp2, tmp2, 4);

                tcg_gen_and_i32(tmp, tmp, tmp2);

            }

            break;

        case 1:

            for (i = 0; i < 3; i ++) {

                tcg_gen_shli_i32(tmp2, tmp2, 8);

                tcg_gen_and_i32(tmp, tmp, tmp2);

            }

            break;

        case 2:

            tcg_gen_shli_i32(tmp2, tmp2, 16);

            tcg_gen_and_i32(tmp, tmp, tmp2);

            break;

        }

        gen_set_nzcv(tmp);

        dead_tmp(tmp2);

        dead_tmp(tmp);

        break;

    case 0x01c: case 0x41c: case 0x81c: case 0xc1c:	/* WACC */

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            gen_helper_iwmmxt_addcb(cpu_M0, cpu_M0);

            break;

        case 1:

            gen_helper_iwmmxt_addcw(cpu_M0, cpu_M0);

            break;

        case 2:

            gen_helper_iwmmxt_addcl(cpu_M0, cpu_M0);

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x115: case 0x515: case 0x915: case 0xd15:	/* TORC */

        if ((insn & 0x000ff00f) != 0x0003f000 || ((insn >> 22) & 3) == 3)

            return 1;

        tmp = iwmmxt_load_creg(ARM_IWMMXT_wCASF);

        tmp2 = new_tmp();

        tcg_gen_mov_i32(tmp2, tmp);

        switch ((insn >> 22) & 3) {

        case 0:

            for (i = 0; i < 7; i ++) {

                tcg_gen_shli_i32(tmp2, tmp2, 4);

                tcg_gen_or_i32(tmp, tmp, tmp2);

            }

            break;

        case 1:

            for (i = 0; i < 3; i ++) {

                tcg_gen_shli_i32(tmp2, tmp2, 8);

                tcg_gen_or_i32(tmp, tmp, tmp2);

            }

            break;

        case 2:

            tcg_gen_shli_i32(tmp2, tmp2, 16);

            tcg_gen_or_i32(tmp, tmp, tmp2);

            break;

        }

        gen_set_nzcv(tmp);

        dead_tmp(tmp2);

        dead_tmp(tmp);

        break;

    case 0x103: case 0x503: case 0x903: case 0xd03:	/* TMOVMSK */

        rd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        if ((insn & 0xf) != 0 || ((insn >> 22) & 3) == 3)

            return 1;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = new_tmp();

        switch ((insn >> 22) & 3) {

        case 0:

            gen_helper_iwmmxt_msbb(tmp, cpu_M0);

            break;

        case 1:

            gen_helper_iwmmxt_msbw(tmp, cpu_M0);

            break;

        case 2:

            gen_helper_iwmmxt_msbl(tmp, cpu_M0);

            break;

        }

        store_reg(s, rd, tmp);

        break;

    case 0x106: case 0x306: case 0x506: case 0x706:	/* WCMPGT */

    case 0x906: case 0xb06: case 0xd06: case 0xf06:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            if (insn & (1 << 21))

                gen_op_iwmmxt_cmpgtsb_M0_wRn(rd1);

            else

                gen_op_iwmmxt_cmpgtub_M0_wRn(rd1);

            break;

        case 1:

            if (insn & (1 << 21))

                gen_op_iwmmxt_cmpgtsw_M0_wRn(rd1);

            else

                gen_op_iwmmxt_cmpgtuw_M0_wRn(rd1);

            break;

        case 2:

            if (insn & (1 << 21))

                gen_op_iwmmxt_cmpgtsl_M0_wRn(rd1);

            else

                gen_op_iwmmxt_cmpgtul_M0_wRn(rd1);

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x00e: case 0x20e: case 0x40e: case 0x60e:	/* WUNPCKEL */

    case 0x80e: case 0xa0e: case 0xc0e: case 0xe0e:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            if (insn & (1 << 21))

                gen_op_iwmmxt_unpacklsb_M0();

            else

                gen_op_iwmmxt_unpacklub_M0();

            break;

        case 1:

            if (insn & (1 << 21))

                gen_op_iwmmxt_unpacklsw_M0();

            else

                gen_op_iwmmxt_unpackluw_M0();

            break;

        case 2:

            if (insn & (1 << 21))

                gen_op_iwmmxt_unpacklsl_M0();

            else

                gen_op_iwmmxt_unpacklul_M0();

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x00c: case 0x20c: case 0x40c: case 0x60c:	/* WUNPCKEH */

    case 0x80c: case 0xa0c: case 0xc0c: case 0xe0c:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            if (insn & (1 << 21))

                gen_op_iwmmxt_unpackhsb_M0();

            else

                gen_op_iwmmxt_unpackhub_M0();

            break;

        case 1:

            if (insn & (1 << 21))

                gen_op_iwmmxt_unpackhsw_M0();

            else

                gen_op_iwmmxt_unpackhuw_M0();

            break;

        case 2:

            if (insn & (1 << 21))

                gen_op_iwmmxt_unpackhsl_M0();

            else

                gen_op_iwmmxt_unpackhul_M0();

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x204: case 0x604: case 0xa04: case 0xe04:	/* WSRL */

    case 0x214: case 0x614: case 0xa14: case 0xe14:

        if (((insn >> 22) & 3) == 0)

            return 1;

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = new_tmp();

        if (gen_iwmmxt_shift(insn, 0xff, tmp)) {

            dead_tmp(tmp);

            return 1;

        }

        switch ((insn >> 22) & 3) {

        case 1:

            gen_helper_iwmmxt_srlw(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 2:

            gen_helper_iwmmxt_srll(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 3:

            gen_helper_iwmmxt_srlq(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        }

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x004: case 0x404: case 0x804: case 0xc04:	/* WSRA */

    case 0x014: case 0x414: case 0x814: case 0xc14:

        if (((insn >> 22) & 3) == 0)

            return 1;

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = new_tmp();

        if (gen_iwmmxt_shift(insn, 0xff, tmp)) {

            dead_tmp(tmp);

            return 1;

        }

        switch ((insn >> 22) & 3) {

        case 1:

            gen_helper_iwmmxt_sraw(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 2:

            gen_helper_iwmmxt_sral(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 3:

            gen_helper_iwmmxt_sraq(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        }

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x104: case 0x504: case 0x904: case 0xd04:	/* WSLL */

    case 0x114: case 0x514: case 0x914: case 0xd14:

        if (((insn >> 22) & 3) == 0)

            return 1;

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = new_tmp();

        if (gen_iwmmxt_shift(insn, 0xff, tmp)) {

            dead_tmp(tmp);

            return 1;

        }

        switch ((insn >> 22) & 3) {

        case 1:

            gen_helper_iwmmxt_sllw(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 2:

            gen_helper_iwmmxt_slll(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 3:

            gen_helper_iwmmxt_sllq(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        }

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x304: case 0x704: case 0xb04: case 0xf04:	/* WROR */

    case 0x314: case 0x714: case 0xb14: case 0xf14:

        if (((insn >> 22) & 3) == 0)

            return 1;

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = new_tmp();

        switch ((insn >> 22) & 3) {

        case 1:

            if (gen_iwmmxt_shift(insn, 0xf, tmp)) {

                dead_tmp(tmp);

                return 1;

            }

            gen_helper_iwmmxt_rorw(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 2:

            if (gen_iwmmxt_shift(insn, 0x1f, tmp)) {

                dead_tmp(tmp);

                return 1;

            }

            gen_helper_iwmmxt_rorl(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        case 3:

            if (gen_iwmmxt_shift(insn, 0x3f, tmp)) {

                dead_tmp(tmp);

                return 1;

            }

            gen_helper_iwmmxt_rorq(cpu_M0, cpu_env, cpu_M0, tmp);

            break;

        }

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x116: case 0x316: case 0x516: case 0x716:	/* WMIN */

    case 0x916: case 0xb16: case 0xd16: case 0xf16:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            if (insn & (1 << 21))

                gen_op_iwmmxt_minsb_M0_wRn(rd1);

            else

                gen_op_iwmmxt_minub_M0_wRn(rd1);

            break;

        case 1:

            if (insn & (1 << 21))

                gen_op_iwmmxt_minsw_M0_wRn(rd1);

            else

                gen_op_iwmmxt_minuw_M0_wRn(rd1);

            break;

        case 2:

            if (insn & (1 << 21))

                gen_op_iwmmxt_minsl_M0_wRn(rd1);

            else

                gen_op_iwmmxt_minul_M0_wRn(rd1);

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x016: case 0x216: case 0x416: case 0x616:	/* WMAX */

    case 0x816: case 0xa16: case 0xc16: case 0xe16:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 0:

            if (insn & (1 << 21))

                gen_op_iwmmxt_maxsb_M0_wRn(rd1);

            else

                gen_op_iwmmxt_maxub_M0_wRn(rd1);

            break;

        case 1:

            if (insn & (1 << 21))

                gen_op_iwmmxt_maxsw_M0_wRn(rd1);

            else

                gen_op_iwmmxt_maxuw_M0_wRn(rd1);

            break;

        case 2:

            if (insn & (1 << 21))

                gen_op_iwmmxt_maxsl_M0_wRn(rd1);

            else

                gen_op_iwmmxt_maxul_M0_wRn(rd1);

            break;

        case 3:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x002: case 0x102: case 0x202: case 0x302:	/* WALIGNI */

    case 0x402: case 0x502: case 0x602: case 0x702:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = tcg_const_i32((insn >> 20) & 3);

        iwmmxt_load_reg(cpu_V1, rd1);

        gen_helper_iwmmxt_align(cpu_M0, cpu_M0, cpu_V1, tmp);

        tcg_temp_free(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    case 0x01a: case 0x11a: case 0x21a: case 0x31a:	/* WSUB */

    case 0x41a: case 0x51a: case 0x61a: case 0x71a:

    case 0x81a: case 0x91a: case 0xa1a: case 0xb1a:

    case 0xc1a: case 0xd1a: case 0xe1a: case 0xf1a:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 20) & 0xf) {

        case 0x0:

            gen_op_iwmmxt_subnb_M0_wRn(rd1);

            break;

        case 0x1:

            gen_op_iwmmxt_subub_M0_wRn(rd1);

            break;

        case 0x3:

            gen_op_iwmmxt_subsb_M0_wRn(rd1);

            break;

        case 0x4:

            gen_op_iwmmxt_subnw_M0_wRn(rd1);

            break;

        case 0x5:

            gen_op_iwmmxt_subuw_M0_wRn(rd1);

            break;

        case 0x7:

            gen_op_iwmmxt_subsw_M0_wRn(rd1);

            break;

        case 0x8:

            gen_op_iwmmxt_subnl_M0_wRn(rd1);

            break;

        case 0x9:

            gen_op_iwmmxt_subul_M0_wRn(rd1);

            break;

        case 0xb:

            gen_op_iwmmxt_subsl_M0_wRn(rd1);

            break;

        default:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x01e: case 0x11e: case 0x21e: case 0x31e:	/* WSHUFH */

    case 0x41e: case 0x51e: case 0x61e: case 0x71e:

    case 0x81e: case 0x91e: case 0xa1e: case 0xb1e:

    case 0xc1e: case 0xd1e: case 0xe1e: case 0xf1e:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        tmp = tcg_const_i32(((insn >> 16) & 0xf0) | (insn & 0x0f));

        gen_helper_iwmmxt_shufh(cpu_M0, cpu_env, cpu_M0, tmp);

        tcg_temp_free(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x018: case 0x118: case 0x218: case 0x318:	/* WADD */

    case 0x418: case 0x518: case 0x618: case 0x718:

    case 0x818: case 0x918: case 0xa18: case 0xb18:

    case 0xc18: case 0xd18: case 0xe18: case 0xf18:

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 20) & 0xf) {

        case 0x0:

            gen_op_iwmmxt_addnb_M0_wRn(rd1);

            break;

        case 0x1:

            gen_op_iwmmxt_addub_M0_wRn(rd1);

            break;

        case 0x3:

            gen_op_iwmmxt_addsb_M0_wRn(rd1);

            break;

        case 0x4:

            gen_op_iwmmxt_addnw_M0_wRn(rd1);

            break;

        case 0x5:

            gen_op_iwmmxt_adduw_M0_wRn(rd1);

            break;

        case 0x7:

            gen_op_iwmmxt_addsw_M0_wRn(rd1);

            break;

        case 0x8:

            gen_op_iwmmxt_addnl_M0_wRn(rd1);

            break;

        case 0x9:

            gen_op_iwmmxt_addul_M0_wRn(rd1);

            break;

        case 0xb:

            gen_op_iwmmxt_addsl_M0_wRn(rd1);

            break;

        default:

            return 1;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x008: case 0x108: case 0x208: case 0x308:	/* WPACK */

    case 0x408: case 0x508: case 0x608: case 0x708:

    case 0x808: case 0x908: case 0xa08: case 0xb08:

    case 0xc08: case 0xd08: case 0xe08: case 0xf08:

        if (!(insn & (1 << 20)) || ((insn >> 22) & 3) == 0)

            return 1;

        wrd = (insn >> 12) & 0xf;

        rd0 = (insn >> 16) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        gen_op_iwmmxt_movq_M0_wRn(rd0);

        switch ((insn >> 22) & 3) {

        case 1:

            if (insn & (1 << 21))

                gen_op_iwmmxt_packsw_M0_wRn(rd1);

            else

                gen_op_iwmmxt_packuw_M0_wRn(rd1);

            break;

        case 2:

            if (insn & (1 << 21))

                gen_op_iwmmxt_packsl_M0_wRn(rd1);

            else

                gen_op_iwmmxt_packul_M0_wRn(rd1);

            break;

        case 3:

            if (insn & (1 << 21))

                gen_op_iwmmxt_packsq_M0_wRn(rd1);

            else

                gen_op_iwmmxt_packuq_M0_wRn(rd1);

            break;

        }

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        gen_op_iwmmxt_set_cup();

        break;

    case 0x201: case 0x203: case 0x205: case 0x207:

    case 0x209: case 0x20b: case 0x20d: case 0x20f:

    case 0x211: case 0x213: case 0x215: case 0x217:

    case 0x219: case 0x21b: case 0x21d: case 0x21f:

        wrd = (insn >> 5) & 0xf;

        rd0 = (insn >> 12) & 0xf;

        rd1 = (insn >> 0) & 0xf;

        if (rd0 == 0xf || rd1 == 0xf)

            return 1;

        gen_op_iwmmxt_movq_M0_wRn(wrd);

        tmp = load_reg(s, rd0);

        tmp2 = load_reg(s, rd1);

        switch ((insn >> 16) & 0xf) {

        case 0x0:					/* TMIA */

            gen_helper_iwmmxt_muladdsl(cpu_M0, cpu_M0, tmp, tmp2);

            break;

        case 0x8:					/* TMIAPH */

            gen_helper_iwmmxt_muladdsw(cpu_M0, cpu_M0, tmp, tmp2);

            break;

        case 0xc: case 0xd: case 0xe: case 0xf:		/* TMIAxy */

            if (insn & (1 << 16))

                tcg_gen_shri_i32(tmp, tmp, 16);

            if (insn & (1 << 17))

                tcg_gen_shri_i32(tmp2, tmp2, 16);

            gen_helper_iwmmxt_muladdswl(cpu_M0, cpu_M0, tmp, tmp2);

            break;

        default:

            dead_tmp(tmp2);

            dead_tmp(tmp);

            return 1;

        }

        dead_tmp(tmp2);

        dead_tmp(tmp);

        gen_op_iwmmxt_movq_wRn_M0(wrd);

        gen_op_iwmmxt_set_mup();

        break;

    default:

        return 1;

    }



    return 0;

}