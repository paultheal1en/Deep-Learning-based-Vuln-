static void disas_sparc_insn(DisasContext * dc)

{

    unsigned int insn, opc, rs1, rs2, rd;



    insn = ldl_code(dc->pc);

    opc = GET_FIELD(insn, 0, 1);



    rd = GET_FIELD(insn, 2, 6);

    switch (opc) {

    case 0:			/* branches/sethi */

	{

	    unsigned int xop = GET_FIELD(insn, 7, 9);

	    int32_t target;

	    switch (xop) {

#ifdef TARGET_SPARC64

	    case 0x1:		/* V9 BPcc */

		{

		    int cc;



		    target = GET_FIELD_SP(insn, 0, 18);

		    target = sign_extend(target, 18);

		    target <<= 2;

		    cc = GET_FIELD_SP(insn, 20, 21);

		    if (cc == 0)

			do_branch(dc, target, insn, 0);

		    else if (cc == 2)

			do_branch(dc, target, insn, 1);

		    else


		    goto jmp_insn;

		}

	    case 0x3:		/* V9 BPr */

		{

		    target = GET_FIELD_SP(insn, 0, 13) | 

                        (GET_FIELD_SP(insn, 20, 21) << 14);

		    target = sign_extend(target, 16);

		    target <<= 2;

		    rs1 = GET_FIELD(insn, 13, 17);

		    gen_movl_reg_T0(rs1);

		    do_branch_reg(dc, target, insn);

		    goto jmp_insn;

		}

	    case 0x5:		/* V9 FBPcc */

		{

		    int cc = GET_FIELD_SP(insn, 20, 21);

                    if (gen_trap_ifnofpu(dc))

                        goto jmp_insn;

		    target = GET_FIELD_SP(insn, 0, 18);

		    target = sign_extend(target, 19);

		    target <<= 2;

		    do_fbranch(dc, target, insn, cc);

		    goto jmp_insn;

		}

#endif

	    case 0x2:		/* BN+x */

		{

		    target = GET_FIELD(insn, 10, 31);

		    target = sign_extend(target, 22);

		    target <<= 2;

		    do_branch(dc, target, insn, 0);

		    goto jmp_insn;

		}

	    case 0x6:		/* FBN+x */

		{

                    if (gen_trap_ifnofpu(dc))

                        goto jmp_insn;

		    target = GET_FIELD(insn, 10, 31);

		    target = sign_extend(target, 22);

		    target <<= 2;

		    do_fbranch(dc, target, insn, 0);

		    goto jmp_insn;

		}

	    case 0x4:		/* SETHI */

#define OPTIM

#if defined(OPTIM)

		if (rd) { // nop

#endif

		    uint32_t value = GET_FIELD(insn, 10, 31);

		    gen_movl_imm_T0(value << 10);

		    gen_movl_T0_reg(rd);

#if defined(OPTIM)

		}

#endif

		break;

	    case 0x0:		/* UNIMPL */

	    default:


	    }

	    break;

	}

	break;

    case 1:

	/*CALL*/ {

	    target_long target = GET_FIELDs(insn, 2, 31) << 2;



#ifdef TARGET_SPARC64

	    if (dc->pc == (uint32_t)dc->pc) {

		gen_op_movl_T0_im(dc->pc);

	    } else {

		gen_op_movq_T0_im64(dc->pc >> 32, dc->pc);

	    }

#else

	    gen_op_movl_T0_im(dc->pc);

#endif

	    gen_movl_T0_reg(15);

	    target += dc->pc;

            gen_mov_pc_npc(dc);

	    dc->npc = target;

	}

	goto jmp_insn;

    case 2:			/* FPU & Logical Operations */

	{

	    unsigned int xop = GET_FIELD(insn, 7, 12);

	    if (xop == 0x3a) {	/* generate trap */

                int cond;



                rs1 = GET_FIELD(insn, 13, 17);

                gen_movl_reg_T0(rs1);

		if (IS_IMM) {

		    rs2 = GET_FIELD(insn, 25, 31);

#if defined(OPTIM)

		    if (rs2 != 0) {

#endif

			gen_movl_simm_T1(rs2);

			gen_op_add_T1_T0();

#if defined(OPTIM)

		    }

#endif

                } else {

                    rs2 = GET_FIELD(insn, 27, 31);

#if defined(OPTIM)

		    if (rs2 != 0) {

#endif

			gen_movl_reg_T1(rs2);

			gen_op_add_T1_T0();

#if defined(OPTIM)

		    }

#endif

                }

                cond = GET_FIELD(insn, 3, 6);

                if (cond == 0x8) {

                    save_state(dc);

                    gen_op_trap_T0();

                } else if (cond != 0) {

#ifdef TARGET_SPARC64

		    /* V9 icc/xcc */

		    int cc = GET_FIELD_SP(insn, 11, 12);

		    flush_T2(dc);

                    save_state(dc);

		    if (cc == 0)

			gen_cond[0][cond]();

		    else if (cc == 2)

			gen_cond[1][cond]();

		    else


#else

		    flush_T2(dc);

                    save_state(dc);

		    gen_cond[0][cond]();

#endif

                    gen_op_trapcc_T0();

                }

                gen_op_next_insn();

                gen_op_movl_T0_0();

                gen_op_exit_tb();

                dc->is_br = 1;

                goto jmp_insn;

            } else if (xop == 0x28) {

                rs1 = GET_FIELD(insn, 13, 17);

                switch(rs1) {

                case 0: /* rdy */

#ifndef TARGET_SPARC64

                case 0x01 ... 0x0e: /* undefined in the SPARCv8

                                       manual, rdy on the microSPARC

                                       II */

                case 0x0f:          /* stbar in the SPARCv8 manual,

                                       rdy on the microSPARC II */

                case 0x10 ... 0x1f: /* implementation-dependent in the

                                       SPARCv8 manual, rdy on the

                                       microSPARC II */

#endif

                    gen_op_movtl_T0_env(offsetof(CPUSPARCState, y));

                    gen_movl_T0_reg(rd);

                    break;

#ifdef TARGET_SPARC64

		case 0x2: /* V9 rdccr */

                    gen_op_rdccr();

                    gen_movl_T0_reg(rd);

                    break;

		case 0x3: /* V9 rdasi */

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, asi));

                    gen_movl_T0_reg(rd);

                    break;

		case 0x4: /* V9 rdtick */

                    gen_op_rdtick();

                    gen_movl_T0_reg(rd);

                    break;

		case 0x5: /* V9 rdpc */

		    if (dc->pc == (uint32_t)dc->pc) {

			gen_op_movl_T0_im(dc->pc);

		    } else {

			gen_op_movq_T0_im64(dc->pc >> 32, dc->pc);

		    }

		    gen_movl_T0_reg(rd);

		    break;

		case 0x6: /* V9 rdfprs */

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, fprs));

                    gen_movl_T0_reg(rd);

                    break;

                case 0xf: /* V9 membar */

                    break; /* no effect */

		case 0x13: /* Graphics Status */

                    if (gen_trap_ifnofpu(dc))

                        goto jmp_insn;

		    gen_op_movtl_T0_env(offsetof(CPUSPARCState, gsr));

                    gen_movl_T0_reg(rd);

                    break;

		case 0x17: /* Tick compare */

		    gen_op_movtl_T0_env(offsetof(CPUSPARCState, tick_cmpr));

                    gen_movl_T0_reg(rd);

                    break;

		case 0x18: /* System tick */

                    gen_op_rdtick(); // XXX

                    gen_movl_T0_reg(rd);

                    break;

		case 0x19: /* System tick compare */

		    gen_op_movtl_T0_env(offsetof(CPUSPARCState, stick_cmpr));

                    gen_movl_T0_reg(rd);

                    break;

		case 0x10: /* Performance Control */

		case 0x11: /* Performance Instrumentation Counter */

		case 0x12: /* Dispatch Control */

		case 0x14: /* Softint set, WO */

		case 0x15: /* Softint clear, WO */

		case 0x16: /* Softint write */

#endif

                default:


                }

#if !defined(CONFIG_USER_ONLY)

#ifndef TARGET_SPARC64

            } else if (xop == 0x29) { /* rdpsr / V9 unimp */

		if (!supervisor(dc))

		    goto priv_insn;

                gen_op_rdpsr();

                gen_movl_T0_reg(rd);

                break;

#endif

            } else if (xop == 0x2a) { /* rdwim / V9 rdpr */

		if (!supervisor(dc))

		    goto priv_insn;

#ifdef TARGET_SPARC64

                rs1 = GET_FIELD(insn, 13, 17);

		switch (rs1) {

		case 0: // tpc

		    gen_op_rdtpc();

		    break;

		case 1: // tnpc

		    gen_op_rdtnpc();

		    break;

		case 2: // tstate

		    gen_op_rdtstate();

		    break;

		case 3: // tt

		    gen_op_rdtt();

		    break;

		case 4: // tick

		    gen_op_rdtick();

		    break;

		case 5: // tba

		    gen_op_movtl_T0_env(offsetof(CPUSPARCState, tbr));

		    break;

		case 6: // pstate

		    gen_op_rdpstate();

		    break;

		case 7: // tl

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, tl));

		    break;

		case 8: // pil

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, psrpil));

		    break;

		case 9: // cwp

		    gen_op_rdcwp();

		    break;

		case 10: // cansave

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, cansave));

		    break;

		case 11: // canrestore

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, canrestore));

		    break;

		case 12: // cleanwin

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, cleanwin));

		    break;

		case 13: // otherwin

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, otherwin));

		    break;

		case 14: // wstate

		    gen_op_movl_T0_env(offsetof(CPUSPARCState, wstate));

		    break;

		case 31: // ver

		    gen_op_movtl_T0_env(offsetof(CPUSPARCState, version));

		    break;

		case 15: // fq

		default:


		}

#else

		gen_op_movl_T0_env(offsetof(CPUSPARCState, wim));

#endif

                gen_movl_T0_reg(rd);

                break;

            } else if (xop == 0x2b) { /* rdtbr / V9 flushw */

#ifdef TARGET_SPARC64

		gen_op_flushw();

#else

		if (!supervisor(dc))

		    goto priv_insn;

		gen_op_movtl_T0_env(offsetof(CPUSPARCState, tbr));

                gen_movl_T0_reg(rd);

#endif

                break;

#endif

	    } else if (xop == 0x34) {	/* FPU Operations */

                if (gen_trap_ifnofpu(dc))

                    goto jmp_insn;

                rs1 = GET_FIELD(insn, 13, 17);

	        rs2 = GET_FIELD(insn, 27, 31);

	        xop = GET_FIELD(insn, 18, 26);

		switch (xop) {

		    case 0x1: /* fmovs */

                	gen_op_load_fpr_FT0(rs2);

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x5: /* fnegs */

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fnegs();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x9: /* fabss */

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fabss();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x29: /* fsqrts */

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fsqrts();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x2a: /* fsqrtd */

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fsqrtd();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x2b: /* fsqrtq */

		        goto nfpu_insn;

		    case 0x41:

                	gen_op_load_fpr_FT0(rs1);

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fadds();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x42:

                	gen_op_load_fpr_DT0(DFPREG(rs1));

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_faddd();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x43: /* faddq */

		        goto nfpu_insn;

		    case 0x45:

                	gen_op_load_fpr_FT0(rs1);

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fsubs();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x46:

                	gen_op_load_fpr_DT0(DFPREG(rs1));

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fsubd();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x47: /* fsubq */

		        goto nfpu_insn;

		    case 0x49:

                	gen_op_load_fpr_FT0(rs1);

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fmuls();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x4a:

                	gen_op_load_fpr_DT0(DFPREG(rs1));

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fmuld();

			gen_op_store_DT0_fpr(rd);

			break;

		    case 0x4b: /* fmulq */

		        goto nfpu_insn;

		    case 0x4d:

                	gen_op_load_fpr_FT0(rs1);

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fdivs();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x4e:

                	gen_op_load_fpr_DT0(DFPREG(rs1));

			gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fdivd();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x4f: /* fdivq */

		        goto nfpu_insn;

		    case 0x69:

                	gen_op_load_fpr_FT0(rs1);

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fsmuld();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x6e: /* fdmulq */

		        goto nfpu_insn;

		    case 0xc4:

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fitos();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0xc6:

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fdtos();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0xc7: /* fqtos */

		        goto nfpu_insn;

		    case 0xc8:

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fitod();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0xc9:

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fstod();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0xcb: /* fqtod */

		        goto nfpu_insn;

		    case 0xcc: /* fitoq */

		        goto nfpu_insn;

		    case 0xcd: /* fstoq */

		        goto nfpu_insn;

		    case 0xce: /* fdtoq */

		        goto nfpu_insn;

		    case 0xd1:

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fstoi();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0xd2:

                	gen_op_load_fpr_DT1(rs2);

			gen_op_fdtoi();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0xd3: /* fqtoi */

		        goto nfpu_insn;

#ifdef TARGET_SPARC64

		    case 0x2: /* V9 fmovd */

                	gen_op_load_fpr_DT0(DFPREG(rs2));

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x6: /* V9 fnegd */

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fnegd();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0xa: /* V9 fabsd */

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fabsd();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x81: /* V9 fstox */

                	gen_op_load_fpr_FT1(rs2);

			gen_op_fstox();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x82: /* V9 fdtox */

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fdtox();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x84: /* V9 fxtos */

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fxtos();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x88: /* V9 fxtod */

                	gen_op_load_fpr_DT1(DFPREG(rs2));

			gen_op_fxtod();

			gen_op_store_DT0_fpr(DFPREG(rd));

			break;

		    case 0x3: /* V9 fmovq */

		    case 0x7: /* V9 fnegq */

		    case 0xb: /* V9 fabsq */

		    case 0x83: /* V9 fqtox */

		    case 0x8c: /* V9 fxtoq */

		        goto nfpu_insn;

#endif

		    default:


		}

	    } else if (xop == 0x35) {	/* FPU Operations */

#ifdef TARGET_SPARC64

		int cond;

#endif

                if (gen_trap_ifnofpu(dc))

                    goto jmp_insn;

                rs1 = GET_FIELD(insn, 13, 17);

	        rs2 = GET_FIELD(insn, 27, 31);

	        xop = GET_FIELD(insn, 18, 26);

#ifdef TARGET_SPARC64

		if ((xop & 0x11f) == 0x005) { // V9 fmovsr

		    cond = GET_FIELD_SP(insn, 14, 17);

		    gen_op_load_fpr_FT0(rd);

		    gen_op_load_fpr_FT1(rs2);

		    rs1 = GET_FIELD(insn, 13, 17);

		    gen_movl_reg_T0(rs1);

		    flush_T2(dc);

		    gen_cond_reg(cond);

		    gen_op_fmovs_cc();

		    gen_op_store_FT0_fpr(rd);

		    break;

		} else if ((xop & 0x11f) == 0x006) { // V9 fmovdr

		    cond = GET_FIELD_SP(insn, 14, 17);

		    gen_op_load_fpr_DT0(rd);

		    gen_op_load_fpr_DT1(rs2);

		    flush_T2(dc);

		    rs1 = GET_FIELD(insn, 13, 17);

		    gen_movl_reg_T0(rs1);

		    gen_cond_reg(cond);

		    gen_op_fmovs_cc();

		    gen_op_store_DT0_fpr(rd);

		    break;

		} else if ((xop & 0x11f) == 0x007) { // V9 fmovqr

		    goto nfpu_insn;

		}

#endif

		switch (xop) {

#ifdef TARGET_SPARC64

		    case 0x001: /* V9 fmovscc %fcc0 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_FT0(rd);

                	gen_op_load_fpr_FT1(rs2);

			flush_T2(dc);

			gen_fcond[0][cond]();

			gen_op_fmovs_cc();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x002: /* V9 fmovdcc %fcc0 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_DT0(rd);

                	gen_op_load_fpr_DT1(rs2);

			flush_T2(dc);

			gen_fcond[0][cond]();

			gen_op_fmovd_cc();

			gen_op_store_DT0_fpr(rd);

			break;

		    case 0x003: /* V9 fmovqcc %fcc0 */

		        goto nfpu_insn;

		    case 0x041: /* V9 fmovscc %fcc1 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_FT0(rd);

                	gen_op_load_fpr_FT1(rs2);

			flush_T2(dc);

			gen_fcond[1][cond]();

			gen_op_fmovs_cc();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x042: /* V9 fmovdcc %fcc1 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_DT0(rd);

                	gen_op_load_fpr_DT1(rs2);

			flush_T2(dc);

			gen_fcond[1][cond]();

			gen_op_fmovd_cc();

			gen_op_store_DT0_fpr(rd);

			break;

		    case 0x043: /* V9 fmovqcc %fcc1 */

		        goto nfpu_insn;

		    case 0x081: /* V9 fmovscc %fcc2 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_FT0(rd);

                	gen_op_load_fpr_FT1(rs2);

			flush_T2(dc);

			gen_fcond[2][cond]();

			gen_op_fmovs_cc();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x082: /* V9 fmovdcc %fcc2 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_DT0(rd);

                	gen_op_load_fpr_DT1(rs2);

			flush_T2(dc);

			gen_fcond[2][cond]();

			gen_op_fmovd_cc();

			gen_op_store_DT0_fpr(rd);

			break;

		    case 0x083: /* V9 fmovqcc %fcc2 */

		        goto nfpu_insn;

		    case 0x0c1: /* V9 fmovscc %fcc3 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_FT0(rd);

                	gen_op_load_fpr_FT1(rs2);

			flush_T2(dc);

			gen_fcond[3][cond]();

			gen_op_fmovs_cc();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x0c2: /* V9 fmovdcc %fcc3 */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_DT0(rd);

                	gen_op_load_fpr_DT1(rs2);

			flush_T2(dc);

			gen_fcond[3][cond]();

			gen_op_fmovd_cc();

			gen_op_store_DT0_fpr(rd);

			break;

		    case 0x0c3: /* V9 fmovqcc %fcc3 */

		        goto nfpu_insn;

		    case 0x101: /* V9 fmovscc %icc */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_FT0(rd);

                	gen_op_load_fpr_FT1(rs2);

			flush_T2(dc);

			gen_cond[0][cond]();

			gen_op_fmovs_cc();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x102: /* V9 fmovdcc %icc */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_DT0(rd);

                	gen_op_load_fpr_DT1(rs2);

			flush_T2(dc);

			gen_cond[0][cond]();

			gen_op_fmovd_cc();

			gen_op_store_DT0_fpr(rd);

			break;

		    case 0x103: /* V9 fmovqcc %icc */

		        goto nfpu_insn;

		    case 0x181: /* V9 fmovscc %xcc */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_FT0(rd);

                	gen_op_load_fpr_FT1(rs2);

			flush_T2(dc);

			gen_cond[1][cond]();

			gen_op_fmovs_cc();

			gen_op_store_FT0_fpr(rd);

			break;

		    case 0x182: /* V9 fmovdcc %xcc */

			cond = GET_FIELD_SP(insn, 14, 17);

                	gen_op_load_fpr_DT0(rd);

                	gen_op_load_fpr_DT1(rs2);

			flush_T2(dc);

			gen_cond[1][cond]();

			gen_op_fmovd_cc();

			gen_op_store_DT0_fpr(rd);

			break;

		    case 0x183: /* V9 fmovqcc %xcc */

		        goto nfpu_insn;

#endif

		    case 0x51: /* V9 %fcc */

                	gen_op_load_fpr_FT0(rs1);

                	gen_op_load_fpr_FT1(rs2);

#ifdef TARGET_SPARC64

			gen_fcmps[rd & 3]();

#else

			gen_op_fcmps();

#endif

			break;

		    case 0x52: /* V9 %fcc */

                	gen_op_load_fpr_DT0(DFPREG(rs1));

                	gen_op_load_fpr_DT1(DFPREG(rs2));

#ifdef TARGET_SPARC64

			gen_fcmpd[rd & 3]();

#else

			gen_op_fcmpd();

#endif

			break;

		    case 0x53: /* fcmpq */

		        goto nfpu_insn;

		    case 0x55: /* fcmpes, V9 %fcc */

                	gen_op_load_fpr_FT0(rs1);

                	gen_op_load_fpr_FT1(rs2);

#ifdef TARGET_SPARC64

			gen_fcmps[rd & 3]();

#else

			gen_op_fcmps(); /* XXX should trap if qNaN or sNaN  */

#endif

			break;

		    case 0x56: /* fcmped, V9 %fcc */

                	gen_op_load_fpr_DT0(DFPREG(rs1));

                	gen_op_load_fpr_DT1(DFPREG(rs2));

#ifdef TARGET_SPARC64

			gen_fcmpd[rd & 3]();

#else

			gen_op_fcmpd(); /* XXX should trap if qNaN or sNaN  */

#endif

			break;

		    case 0x57: /* fcmpeq */

		        goto nfpu_insn;

		    default:


		}

#if defined(OPTIM)

	    } else if (xop == 0x2) {

		// clr/mov shortcut



                rs1 = GET_FIELD(insn, 13, 17);

		if (rs1 == 0) {

		    // or %g0, x, y -> mov T1, x; mov y, T1

		    if (IS_IMM) {	/* immediate */

			rs2 = GET_FIELDs(insn, 19, 31);

			gen_movl_simm_T1(rs2);

		    } else {		/* register */

			rs2 = GET_FIELD(insn, 27, 31);

			gen_movl_reg_T1(rs2);

		    }

		    gen_movl_T1_reg(rd);

		} else {

		    gen_movl_reg_T0(rs1);

		    if (IS_IMM) {	/* immediate */

			// or x, #0, y -> mov T1, x; mov y, T1

			rs2 = GET_FIELDs(insn, 19, 31);

			if (rs2 != 0) {

			    gen_movl_simm_T1(rs2);

			    gen_op_or_T1_T0();

			}

		    } else {		/* register */

			// or x, %g0, y -> mov T1, x; mov y, T1

			rs2 = GET_FIELD(insn, 27, 31);

			if (rs2 != 0) {

			    gen_movl_reg_T1(rs2);

			    gen_op_or_T1_T0();

			}

		    }

		    gen_movl_T0_reg(rd);

		}

#endif

#ifdef TARGET_SPARC64

	    } else if (xop == 0x25) { /* sll, V9 sllx ( == sll) */

                rs1 = GET_FIELD(insn, 13, 17);

		gen_movl_reg_T0(rs1);

		if (IS_IMM) {	/* immediate */

                    rs2 = GET_FIELDs(insn, 20, 31);

                    gen_movl_simm_T1(rs2);

                } else {		/* register */

                    rs2 = GET_FIELD(insn, 27, 31);

                    gen_movl_reg_T1(rs2);

                }

		gen_op_sll();

		gen_movl_T0_reg(rd);

	    } else if (xop == 0x26) { /* srl, V9 srlx */

                rs1 = GET_FIELD(insn, 13, 17);

		gen_movl_reg_T0(rs1);

		if (IS_IMM) {	/* immediate */

                    rs2 = GET_FIELDs(insn, 20, 31);

                    gen_movl_simm_T1(rs2);

                } else {		/* register */

                    rs2 = GET_FIELD(insn, 27, 31);

                    gen_movl_reg_T1(rs2);

                }

		if (insn & (1 << 12))

		    gen_op_srlx();

		else

		    gen_op_srl();

		gen_movl_T0_reg(rd);

	    } else if (xop == 0x27) { /* sra, V9 srax */

                rs1 = GET_FIELD(insn, 13, 17);

		gen_movl_reg_T0(rs1);

		if (IS_IMM) {	/* immediate */

                    rs2 = GET_FIELDs(insn, 20, 31);

                    gen_movl_simm_T1(rs2);

                } else {		/* register */

                    rs2 = GET_FIELD(insn, 27, 31);

                    gen_movl_reg_T1(rs2);

                }

		if (insn & (1 << 12))

		    gen_op_srax();

		else

		    gen_op_sra();

		gen_movl_T0_reg(rd);

#endif

            } else if (xop < 0x36) {

                rs1 = GET_FIELD(insn, 13, 17);

		gen_movl_reg_T0(rs1);

		if (IS_IMM) {	/* immediate */

                    rs2 = GET_FIELDs(insn, 19, 31);

                    gen_movl_simm_T1(rs2);

                } else {		/* register */

                    rs2 = GET_FIELD(insn, 27, 31);

                    gen_movl_reg_T1(rs2);

                }

                if (xop < 0x20) {

                    switch (xop & ~0x10) {

                    case 0x0:

                        if (xop & 0x10)

                            gen_op_add_T1_T0_cc();

                        else

                            gen_op_add_T1_T0();

                        break;

                    case 0x1:

                        gen_op_and_T1_T0();

                        if (xop & 0x10)

                            gen_op_logic_T0_cc();

                        break;

                    case 0x2:

			gen_op_or_T1_T0();

			if (xop & 0x10)

			    gen_op_logic_T0_cc();

			break;

                    case 0x3:

                        gen_op_xor_T1_T0();

                        if (xop & 0x10)

                            gen_op_logic_T0_cc();

                        break;

                    case 0x4:

                        if (xop & 0x10)

                            gen_op_sub_T1_T0_cc();

                        else

                            gen_op_sub_T1_T0();

                        break;

                    case 0x5:

                        gen_op_andn_T1_T0();

                        if (xop & 0x10)

                            gen_op_logic_T0_cc();

                        break;

                    case 0x6:

                        gen_op_orn_T1_T0();

                        if (xop & 0x10)

                            gen_op_logic_T0_cc();

                        break;

                    case 0x7:

                        gen_op_xnor_T1_T0();

                        if (xop & 0x10)

                            gen_op_logic_T0_cc();

                        break;

                    case 0x8:

                        if (xop & 0x10)

                            gen_op_addx_T1_T0_cc();

                        else

                            gen_op_addx_T1_T0();

                        break;

#ifdef TARGET_SPARC64

		    case 0x9: /* V9 mulx */

                        gen_op_mulx_T1_T0();

                        break;

#endif

                    case 0xa:

                        gen_op_umul_T1_T0();

                        if (xop & 0x10)

                            gen_op_logic_T0_cc();

                        break;

                    case 0xb:

                        gen_op_smul_T1_T0();

                        if (xop & 0x10)

                            gen_op_logic_T0_cc();

                        break;

                    case 0xc:

                        if (xop & 0x10)

                            gen_op_subx_T1_T0_cc();

                        else

                            gen_op_subx_T1_T0();

                        break;

#ifdef TARGET_SPARC64

		    case 0xd: /* V9 udivx */

                        gen_op_udivx_T1_T0();

                        break;

#endif

                    case 0xe:

                        gen_op_udiv_T1_T0();

                        if (xop & 0x10)

                            gen_op_div_cc();

                        break;

                    case 0xf:

                        gen_op_sdiv_T1_T0();

                        if (xop & 0x10)

                            gen_op_div_cc();

                        break;

                    default:


                    }

		    gen_movl_T0_reg(rd);

                } else {

                    switch (xop) {

		    case 0x20: /* taddcc */

			gen_op_tadd_T1_T0_cc();

		        gen_movl_T0_reg(rd);

			break;

		    case 0x21: /* tsubcc */

			gen_op_tsub_T1_T0_cc();

		        gen_movl_T0_reg(rd);

			break;

		    case 0x22: /* taddcctv */

			gen_op_tadd_T1_T0_ccTV();

		        gen_movl_T0_reg(rd);

			break;

		    case 0x23: /* tsubcctv */

			gen_op_tsub_T1_T0_ccTV();

		        gen_movl_T0_reg(rd);

			break;

                    case 0x24: /* mulscc */

                        gen_op_mulscc_T1_T0();

                        gen_movl_T0_reg(rd);

                        break;

#ifndef TARGET_SPARC64

                    case 0x25:	/* sll */

			gen_op_sll();

                        gen_movl_T0_reg(rd);

                        break;

                    case 0x26:  /* srl */

			gen_op_srl();

                        gen_movl_T0_reg(rd);

                        break;

                    case 0x27:  /* sra */

			gen_op_sra();

                        gen_movl_T0_reg(rd);

                        break;

#endif

                    case 0x30:

                        {

                            switch(rd) {

                            case 0: /* wry */

				gen_op_xor_T1_T0();

				gen_op_movtl_env_T0(offsetof(CPUSPARCState, y));

                                break;

#ifndef TARGET_SPARC64

                            case 0x01 ... 0x0f: /* undefined in the

                                                   SPARCv8 manual, nop

                                                   on the microSPARC

                                                   II */

                            case 0x10 ... 0x1f: /* implementation-dependent

                                                   in the SPARCv8

                                                   manual, nop on the

                                                   microSPARC II */

                                break;

#else

			    case 0x2: /* V9 wrccr */

                                gen_op_wrccr();

				break;

			    case 0x3: /* V9 wrasi */

				gen_op_movl_env_T0(offsetof(CPUSPARCState, asi));

				break;

			    case 0x6: /* V9 wrfprs */

				gen_op_movl_env_T0(offsetof(CPUSPARCState, fprs));

				break;

			    case 0xf: /* V9 sir, nop if user */

#if !defined(CONFIG_USER_ONLY)

				if (supervisor(dc))

				    gen_op_sir();

#endif

				break;

			    case 0x13: /* Graphics Status */

                                if (gen_trap_ifnofpu(dc))

                                    goto jmp_insn;

				gen_op_movtl_env_T0(offsetof(CPUSPARCState, gsr));

				break;

			    case 0x17: /* Tick compare */

#if !defined(CONFIG_USER_ONLY)

				if (!supervisor(dc))


#endif

				gen_op_movtl_env_T0(offsetof(CPUSPARCState, tick_cmpr));

				break;

			    case 0x18: /* System tick */

#if !defined(CONFIG_USER_ONLY)

				if (!supervisor(dc))


#endif

				gen_op_movtl_env_T0(offsetof(CPUSPARCState, stick_cmpr));

				break;

			    case 0x19: /* System tick compare */

#if !defined(CONFIG_USER_ONLY)

				if (!supervisor(dc))


#endif

				gen_op_movtl_env_T0(offsetof(CPUSPARCState, stick_cmpr));

				break;



			    case 0x10: /* Performance Control */

			    case 0x11: /* Performance Instrumentation Counter */

			    case 0x12: /* Dispatch Control */

			    case 0x14: /* Softint set */

			    case 0x15: /* Softint clear */

			    case 0x16: /* Softint write */

#endif

                            default:


                            }

                        }

                        break;

#if !defined(CONFIG_USER_ONLY)

                    case 0x31: /* wrpsr, V9 saved, restored */

                        {

			    if (!supervisor(dc))

				goto priv_insn;

#ifdef TARGET_SPARC64

			    switch (rd) {

			    case 0:

				gen_op_saved();

				break;

			    case 1:

				gen_op_restored();

				break;

			    default:


                            }

#else

                            gen_op_xor_T1_T0();

                            gen_op_wrpsr();

                            save_state(dc);

                            gen_op_next_insn();

			    gen_op_movl_T0_0();

			    gen_op_exit_tb();

			    dc->is_br = 1;

#endif

                        }

                        break;

                    case 0x32: /* wrwim, V9 wrpr */

                        {

			    if (!supervisor(dc))

				goto priv_insn;

                            gen_op_xor_T1_T0();

#ifdef TARGET_SPARC64

			    switch (rd) {

			    case 0: // tpc

				gen_op_wrtpc();

				break;

			    case 1: // tnpc

				gen_op_wrtnpc();

				break;

			    case 2: // tstate

				gen_op_wrtstate();

				break;

			    case 3: // tt

				gen_op_wrtt();

				break;

			    case 4: // tick

				gen_op_wrtick();

				break;

			    case 5: // tba

				gen_op_movtl_env_T0(offsetof(CPUSPARCState, tbr));

				break;

			    case 6: // pstate

				gen_op_wrpstate();

                                save_state(dc);

                                gen_op_next_insn();

                                gen_op_movl_T0_0();

                                gen_op_exit_tb();

                                dc->is_br = 1;

				break;

			    case 7: // tl

				gen_op_movl_env_T0(offsetof(CPUSPARCState, tl));

				break;

			    case 8: // pil

				gen_op_movl_env_T0(offsetof(CPUSPARCState, psrpil));

				break;

			    case 9: // cwp

				gen_op_wrcwp();

				break;

			    case 10: // cansave

				gen_op_movl_env_T0(offsetof(CPUSPARCState, cansave));

				break;

			    case 11: // canrestore

				gen_op_movl_env_T0(offsetof(CPUSPARCState, canrestore));

				break;

			    case 12: // cleanwin

				gen_op_movl_env_T0(offsetof(CPUSPARCState, cleanwin));

				break;

			    case 13: // otherwin

				gen_op_movl_env_T0(offsetof(CPUSPARCState, otherwin));

				break;

			    case 14: // wstate

				gen_op_movl_env_T0(offsetof(CPUSPARCState, wstate));

				break;

			    default:


			    }

#else

			    gen_op_wrwim();

#endif

                        }

                        break;

#ifndef TARGET_SPARC64

                    case 0x33: /* wrtbr, V9 unimp */

                        {

			    if (!supervisor(dc))

				goto priv_insn;

                            gen_op_xor_T1_T0();

			    gen_op_movtl_env_T0(offsetof(CPUSPARCState, tbr));

                        }

                        break;

#endif

#endif

#ifdef TARGET_SPARC64

		    case 0x2c: /* V9 movcc */

			{

			    int cc = GET_FIELD_SP(insn, 11, 12);

			    int cond = GET_FIELD_SP(insn, 14, 17);

			    if (IS_IMM) {	/* immediate */

				rs2 = GET_FIELD_SPs(insn, 0, 10);

				gen_movl_simm_T1(rs2);

			    }

			    else {

				rs2 = GET_FIELD_SP(insn, 0, 4);

				gen_movl_reg_T1(rs2);

			    }

			    gen_movl_reg_T0(rd);

			    flush_T2(dc);

			    if (insn & (1 << 18)) {

				if (cc == 0)

				    gen_cond[0][cond]();

				else if (cc == 2)

				    gen_cond[1][cond]();

				else


			    } else {

				gen_fcond[cc][cond]();

			    }

			    gen_op_mov_cc();

			    gen_movl_T0_reg(rd);

			    break;

			}

		    case 0x2d: /* V9 sdivx */

                        gen_op_sdivx_T1_T0();

			gen_movl_T0_reg(rd);

                        break;

		    case 0x2e: /* V9 popc */

			{

			    if (IS_IMM) {	/* immediate */

				rs2 = GET_FIELD_SPs(insn, 0, 12);

				gen_movl_simm_T1(rs2);

				// XXX optimize: popc(constant)

			    }

			    else {

				rs2 = GET_FIELD_SP(insn, 0, 4);

				gen_movl_reg_T1(rs2);

			    }

			    gen_op_popc();

			    gen_movl_T0_reg(rd);

			}

		    case 0x2f: /* V9 movr */

			{

			    int cond = GET_FIELD_SP(insn, 10, 12);

			    rs1 = GET_FIELD(insn, 13, 17);

			    flush_T2(dc);

			    gen_movl_reg_T0(rs1);

			    gen_cond_reg(cond);

			    if (IS_IMM) {	/* immediate */

				rs2 = GET_FIELD_SPs(insn, 0, 10);

				gen_movl_simm_T1(rs2);

			    }

			    else {

				rs2 = GET_FIELD_SP(insn, 0, 4);

				gen_movl_reg_T1(rs2);

			    }

			    gen_movl_reg_T0(rd);

			    gen_op_mov_cc();

			    gen_movl_T0_reg(rd);

			    break;

			}

		    case 0x36: /* UltraSparc shutdown, VIS */

			{

			    int opf = GET_FIELD_SP(insn, 5, 13);

                            rs1 = GET_FIELD(insn, 13, 17);

                            rs2 = GET_FIELD(insn, 27, 31);



                            switch (opf) {

                            case 0x018: /* VIS I alignaddr */

                                if (gen_trap_ifnofpu(dc))

                                    goto jmp_insn;

                                gen_movl_reg_T0(rs1);

                                gen_movl_reg_T1(rs2);

                                gen_op_alignaddr();

                                gen_movl_T0_reg(rd);

                                break;

                            case 0x01a: /* VIS I alignaddrl */

                                if (gen_trap_ifnofpu(dc))

                                    goto jmp_insn;

                                // XXX

                                break;

                            case 0x048: /* VIS I faligndata */

                                if (gen_trap_ifnofpu(dc))

                                    goto jmp_insn;

                                gen_op_load_fpr_DT0(rs1);

                                gen_op_load_fpr_DT1(rs2);

                                gen_op_faligndata();

                                gen_op_store_DT0_fpr(rd);

                                break;

                            default:


                            }

                            break;

			}

#endif

		    default:


		    }

		}

            } else if (xop == 0x36 || xop == 0x37) { /* CPop1 & CPop2,

                                                        V9 impdep1 &

                                                        impdep2 */

#ifdef TARGET_SPARC64


#else

	        goto ncp_insn;

#endif

#ifdef TARGET_SPARC64

	    } else if (xop == 0x39) { /* V9 return */

                rs1 = GET_FIELD(insn, 13, 17);

		gen_movl_reg_T0(rs1);

                if (IS_IMM) {	/* immediate */

		    rs2 = GET_FIELDs(insn, 19, 31);

#if defined(OPTIM)

		    if (rs2) {

#endif

			gen_movl_simm_T1(rs2);

			gen_op_add_T1_T0();

#if defined(OPTIM)

		    }

#endif

                } else {		/* register */

                    rs2 = GET_FIELD(insn, 27, 31);

#if defined(OPTIM)

		    if (rs2) {

#endif

			gen_movl_reg_T1(rs2);

			gen_op_add_T1_T0();

#if defined(OPTIM)

		    }

#endif

                }

		gen_op_restore();

		gen_mov_pc_npc(dc);

		gen_op_movl_npc_T0();

		dc->npc = DYNAMIC_PC;

		goto jmp_insn;

#endif

	    } else {

                rs1 = GET_FIELD(insn, 13, 17);

		gen_movl_reg_T0(rs1);

                if (IS_IMM) {	/* immediate */

		    rs2 = GET_FIELDs(insn, 19, 31);

#if defined(OPTIM)

		    if (rs2) {

#endif

			gen_movl_simm_T1(rs2);

			gen_op_add_T1_T0();

#if defined(OPTIM)

		    }

#endif

                } else {		/* register */

                    rs2 = GET_FIELD(insn, 27, 31);

#if defined(OPTIM)

		    if (rs2) {

#endif

			gen_movl_reg_T1(rs2);

			gen_op_add_T1_T0();

#if defined(OPTIM)

		    }

#endif

                }

		switch (xop) {

		case 0x38:	/* jmpl */

		    {

			if (rd != 0) {

#ifdef TARGET_SPARC64

                            if (dc->pc == (uint32_t)dc->pc) {

                                gen_op_movl_T1_im(dc->pc);

                            } else {

                                gen_op_movq_T1_im64(dc->pc >> 32, dc->pc);

                            }

#else

			    gen_op_movl_T1_im(dc->pc);

#endif

			    gen_movl_T1_reg(rd);

			}

                        gen_mov_pc_npc(dc);

			gen_op_movl_npc_T0();

			dc->npc = DYNAMIC_PC;

		    }

		    goto jmp_insn;

#if !defined(CONFIG_USER_ONLY) && !defined(TARGET_SPARC64)

		case 0x39:	/* rett, V9 return */

		    {

			if (!supervisor(dc))

			    goto priv_insn;

                        gen_mov_pc_npc(dc);

			gen_op_movl_npc_T0();

			dc->npc = DYNAMIC_PC;

			gen_op_rett();

		    }

		    goto jmp_insn;

#endif

		case 0x3b: /* flush */

		    gen_op_flush_T0();

		    break;

		case 0x3c:	/* save */

		    save_state(dc);

		    gen_op_save();

		    gen_movl_T0_reg(rd);

		    break;

		case 0x3d:	/* restore */

		    save_state(dc);

		    gen_op_restore();

		    gen_movl_T0_reg(rd);

		    break;

#if !defined(CONFIG_USER_ONLY) && defined(TARGET_SPARC64)

		case 0x3e:      /* V9 done/retry */

		    {

			switch (rd) {

			case 0:

			    if (!supervisor(dc))

				goto priv_insn;

			    dc->npc = DYNAMIC_PC;

			    dc->pc = DYNAMIC_PC;

			    gen_op_done();

			    goto jmp_insn;

			case 1:

			    if (!supervisor(dc))

				goto priv_insn;

			    dc->npc = DYNAMIC_PC;

			    dc->pc = DYNAMIC_PC;

			    gen_op_retry();

			    goto jmp_insn;

			default:


			}

		    }

		    break;

#endif

		default:


		}

            }

	    break;

	}

	break;

    case 3:			/* load/store instructions */

	{

	    unsigned int xop = GET_FIELD(insn, 7, 12);

	    rs1 = GET_FIELD(insn, 13, 17);

	    gen_movl_reg_T0(rs1);

	    if (IS_IMM) {	/* immediate */

		rs2 = GET_FIELDs(insn, 19, 31);

#if defined(OPTIM)

		if (rs2 != 0) {

#endif

		    gen_movl_simm_T1(rs2);

		    gen_op_add_T1_T0();

#if defined(OPTIM)

		}

#endif

	    } else {		/* register */

		rs2 = GET_FIELD(insn, 27, 31);

#if defined(OPTIM)

		if (rs2 != 0) {

#endif

		    gen_movl_reg_T1(rs2);

		    gen_op_add_T1_T0();

#if defined(OPTIM)

		}

#endif

	    }

	    if (xop < 4 || (xop > 7 && xop < 0x14 && xop != 0x0e) || \

		    (xop > 0x17 && xop < 0x1d ) || \

		    (xop > 0x2c && xop < 0x33) || xop == 0x1f) {

		switch (xop) {

		case 0x0:	/* load word */

		    gen_op_ldst(ld);

		    break;

		case 0x1:	/* load unsigned byte */

		    gen_op_ldst(ldub);

		    break;

		case 0x2:	/* load unsigned halfword */

		    gen_op_ldst(lduh);

		    break;

		case 0x3:	/* load double word */



		    gen_op_ldst(ldd);

		    gen_movl_T0_reg(rd + 1);

		    break;

		case 0x9:	/* load signed byte */

		    gen_op_ldst(ldsb);

		    break;

		case 0xa:	/* load signed halfword */

		    gen_op_ldst(ldsh);

		    break;

		case 0xd:	/* ldstub -- XXX: should be atomically */

		    gen_op_ldst(ldstub);

		    break;

		case 0x0f:	/* swap register with memory. Also atomically */

		    gen_movl_reg_T1(rd);

		    gen_op_ldst(swap);

		    break;

#if !defined(CONFIG_USER_ONLY) || defined(TARGET_SPARC64)

		case 0x10:	/* load word alternate */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_lda(insn, 1, 4, 0);

		    break;

		case 0x11:	/* load unsigned byte alternate */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_lduba(insn, 1, 1, 0);

		    break;

		case 0x12:	/* load unsigned halfword alternate */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_lduha(insn, 1, 2, 0);

		    break;

		case 0x13:	/* load double word alternate */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif



		    gen_op_ldda(insn, 1, 8, 0);

		    gen_movl_T0_reg(rd + 1);

		    break;

		case 0x19:	/* load signed byte alternate */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_ldsba(insn, 1, 1, 1);

		    break;

		case 0x1a:	/* load signed halfword alternate */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_ldsha(insn, 1, 2 ,1);

		    break;

		case 0x1d:	/* ldstuba -- XXX: should be atomically */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_ldstuba(insn, 1, 1, 0);

		    break;

		case 0x1f:	/* swap reg with alt. memory. Also atomically */

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_movl_reg_T1(rd);

		    gen_op_swapa(insn, 1, 4, 0);

		    break;



#ifndef TARGET_SPARC64

		case 0x30: /* ldc */

		case 0x31: /* ldcsr */

		case 0x33: /* lddc */

		case 0x34: /* stc */

		case 0x35: /* stcsr */

		case 0x36: /* stdcq */

		case 0x37: /* stdc */

		    goto ncp_insn;

		    break;

                    /* avoid warnings */

                    (void) &gen_op_stfa;

                    (void) &gen_op_stdfa;

                    (void) &gen_op_ldfa;

                    (void) &gen_op_lddfa;

#else

#if !defined(CONFIG_USER_ONLY)

		    (void) &gen_op_cas;

		    (void) &gen_op_casx;

#endif

#endif

#endif

#ifdef TARGET_SPARC64

		case 0x08: /* V9 ldsw */

		    gen_op_ldst(ldsw);

		    break;

		case 0x0b: /* V9 ldx */

		    gen_op_ldst(ldx);

		    break;

		case 0x18: /* V9 ldswa */

		    gen_op_ldswa(insn, 1, 4, 1);

		    break;

		case 0x1b: /* V9 ldxa */

		    gen_op_ldxa(insn, 1, 8, 0);

		    break;

		case 0x2d: /* V9 prefetch, no effect */

		    goto skip_move;

		case 0x30: /* V9 ldfa */

		    gen_op_ldfa(insn, 1, 8, 0); // XXX

		    break;

		case 0x33: /* V9 lddfa */

		    gen_op_lddfa(insn, 1, 8, 0); // XXX



		    break;

		case 0x3d: /* V9 prefetcha, no effect */

		    goto skip_move;

		case 0x32: /* V9 ldqfa */

		    goto nfpu_insn;

#endif

		default:


		}

		gen_movl_T1_reg(rd);

#ifdef TARGET_SPARC64

	    skip_move: ;

#endif

	    } else if (xop >= 0x20 && xop < 0x24) {

                if (gen_trap_ifnofpu(dc))

                    goto jmp_insn;

		switch (xop) {

		case 0x20:	/* load fpreg */

		    gen_op_ldst(ldf);

		    gen_op_store_FT0_fpr(rd);

		    break;

		case 0x21:	/* load fsr */

		    gen_op_ldst(ldf);

		    gen_op_ldfsr();

		    break;

		case 0x22:      /* load quad fpreg */

		    goto nfpu_insn;

		case 0x23:	/* load double fpreg */

		    gen_op_ldst(lddf);

		    gen_op_store_DT0_fpr(DFPREG(rd));

		    break;

		default:


		}

	    } else if (xop < 8 || (xop >= 0x14 && xop < 0x18) || \

		       xop == 0xe || xop == 0x1e) {

		gen_movl_reg_T1(rd);

		switch (xop) {

		case 0x4:

		    gen_op_ldst(st);

		    break;

		case 0x5:

		    gen_op_ldst(stb);

		    break;

		case 0x6:

		    gen_op_ldst(sth);

		    break;

		case 0x7:



                    flush_T2(dc);

		    gen_movl_reg_T2(rd + 1);

		    gen_op_ldst(std);

		    break;

#if !defined(CONFIG_USER_ONLY) || defined(TARGET_SPARC64)

		case 0x14:

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_sta(insn, 0, 4, 0);

                    break;

		case 0x15:

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_stba(insn, 0, 1, 0);

                    break;

		case 0x16:

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif

		    gen_op_stha(insn, 0, 2, 0);

                    break;

		case 0x17:

#ifndef TARGET_SPARC64

		    if (IS_IMM)


		    if (!supervisor(dc))

			goto priv_insn;

#endif



                    flush_T2(dc);

		    gen_movl_reg_T2(rd + 1);

		    gen_op_stda(insn, 0, 8, 0);

                    break;

#endif

#ifdef TARGET_SPARC64

		case 0x0e: /* V9 stx */

		    gen_op_ldst(stx);

		    break;

		case 0x1e: /* V9 stxa */

		    gen_op_stxa(insn, 0, 8, 0); // XXX

		    break;

#endif

		default:


		}

	    } else if (xop > 0x23 && xop < 0x28) {

                if (gen_trap_ifnofpu(dc))

                    goto jmp_insn;

		switch (xop) {

		case 0x24:

                    gen_op_load_fpr_FT0(rd);

		    gen_op_ldst(stf);

		    break;

		case 0x25: /* stfsr, V9 stxfsr */

		    gen_op_stfsr();

		    gen_op_ldst(stf);

		    break;

		case 0x26: /* stdfq */

		    goto nfpu_insn;

		case 0x27:

                    gen_op_load_fpr_DT0(DFPREG(rd));

		    gen_op_ldst(stdf);

		    break;

		default:


		}

	    } else if (xop > 0x33 && xop < 0x3f) {

#ifdef TARGET_SPARC64

		switch (xop) {

		case 0x34: /* V9 stfa */

		    gen_op_stfa(insn, 0, 0, 0); // XXX

		    break;

		case 0x37: /* V9 stdfa */

		    gen_op_stdfa(insn, 0, 0, 0); // XXX

		    break;

		case 0x3c: /* V9 casa */

		    gen_op_casa(insn, 0, 4, 0); // XXX

		    break;

		case 0x3e: /* V9 casxa */

		    gen_op_casxa(insn, 0, 8, 0); // XXX

		    break;

		case 0x36: /* V9 stqfa */

		    goto nfpu_insn;

		default:


		}

#else


#endif

            }

	    else


	}

	break;

    }

    /* default case for non jump instructions */

    if (dc->npc == DYNAMIC_PC) {

	dc->pc = DYNAMIC_PC;

	gen_op_next_insn();

    } else if (dc->npc == JUMP_PC) {

        /* we can do a static jump */

        gen_branch2(dc, (long)dc->tb, dc->jump_pc[0], dc->jump_pc[1]);

        dc->is_br = 1;

    } else {

	dc->pc = dc->npc;

	dc->npc = dc->npc + 4;

    }

 jmp_insn:

    return;

 illegal_insn:

    save_state(dc);

    gen_op_exception(TT_ILL_INSN);

    dc->is_br = 1;

    return;

#if !defined(CONFIG_USER_ONLY)

 priv_insn:

    save_state(dc);

    gen_op_exception(TT_PRIV_INSN);

    dc->is_br = 1;

    return;

#endif

 nfpu_insn:

    save_state(dc);

    gen_op_fpexception_im(FSR_FTT_UNIMPFPOP);

    dc->is_br = 1;

    return;

#ifndef TARGET_SPARC64

 ncp_insn:

    save_state(dc);

    gen_op_exception(TT_NCP_INSN);

    dc->is_br = 1;

    return;

#endif

}