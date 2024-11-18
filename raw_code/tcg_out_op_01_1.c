static inline void tcg_out_op(TCGContext *s, TCGOpcode opc, const TCGArg *args,

                              const int *const_args)

{

    int c;



    switch (opc) {

    case INDEX_op_exit_tb:

        tcg_out_movi(s, TCG_TYPE_PTR, TCG_REG_I0, args[0]);

        tcg_out32(s, JMPL | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_I7) |

                  INSN_IMM13(8));

        tcg_out32(s, RESTORE | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_G0) |

                      INSN_RS2(TCG_REG_G0));

        break;

    case INDEX_op_goto_tb:

        if (s->tb_jmp_offset) {

            /* direct jump method */

            tcg_out_sethi(s, TCG_REG_I5, args[0] & 0xffffe000);

            tcg_out32(s, JMPL | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_I5) |

                      INSN_IMM13((args[0] & 0x1fff)));

            s->tb_jmp_offset[args[0]] = s->code_ptr - s->code_buf;

        } else {

            /* indirect jump method */

            tcg_out_ld_ptr(s, TCG_REG_I5, (tcg_target_long)(s->tb_next + args[0]));

            tcg_out32(s, JMPL | INSN_RD(TCG_REG_G0) | INSN_RS1(TCG_REG_I5) |

                      INSN_RS2(TCG_REG_G0));

        }

        tcg_out_nop(s);

        s->tb_next_offset[args[0]] = s->code_ptr - s->code_buf;

        break;

    case INDEX_op_call:

        if (const_args[0])

            tcg_out32(s, CALL | ((((tcg_target_ulong)args[0]

                                   - (tcg_target_ulong)s->code_ptr) >> 2)

                                 & 0x3fffffff));

        else {

            tcg_out_ld_ptr(s, TCG_REG_I5,

                           (tcg_target_long)(s->tb_next + args[0]));

            tcg_out32(s, JMPL | INSN_RD(TCG_REG_O7) | INSN_RS1(TCG_REG_I5) |

                      INSN_RS2(TCG_REG_G0));

        }

        /* delay slot */

        tcg_out_nop(s);

        break;

    case INDEX_op_jmp:

    case INDEX_op_br:

        tcg_out_branch_i32(s, COND_A, args[0]);

        tcg_out_nop(s);

        break;

    case INDEX_op_movi_i32:

        tcg_out_movi(s, TCG_TYPE_I32, args[0], (uint32_t)args[1]);

        break;



#if TCG_TARGET_REG_BITS == 64

#define OP_32_64(x)                             \

        glue(glue(case INDEX_op_, x), _i32):    \

        glue(glue(case INDEX_op_, x), _i64)

#else

#define OP_32_64(x)                             \

        glue(glue(case INDEX_op_, x), _i32)

#endif

    OP_32_64(ld8u):

        tcg_out_ldst(s, args[0], args[1], args[2], LDUB);

        break;

    OP_32_64(ld8s):

        tcg_out_ldst(s, args[0], args[1], args[2], LDSB);

        break;

    OP_32_64(ld16u):

        tcg_out_ldst(s, args[0], args[1], args[2], LDUH);

        break;

    OP_32_64(ld16s):

        tcg_out_ldst(s, args[0], args[1], args[2], LDSH);

        break;

    case INDEX_op_ld_i32:

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_ld32u_i64:

#endif

        tcg_out_ldst(s, args[0], args[1], args[2], LDUW);

        break;

    OP_32_64(st8):

        tcg_out_ldst(s, args[0], args[1], args[2], STB);

        break;

    OP_32_64(st16):

        tcg_out_ldst(s, args[0], args[1], args[2], STH);

        break;

    case INDEX_op_st_i32:

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_st32_i64:

#endif

        tcg_out_ldst(s, args[0], args[1], args[2], STW);

        break;

    OP_32_64(add):

        c = ARITH_ADD;

        goto gen_arith;

    OP_32_64(sub):

        c = ARITH_SUB;

        goto gen_arith;

    OP_32_64(and):

        c = ARITH_AND;

        goto gen_arith;

    OP_32_64(andc):

        c = ARITH_ANDN;

        goto gen_arith;

    OP_32_64(or):

        c = ARITH_OR;

        goto gen_arith;

    OP_32_64(orc):

        c = ARITH_ORN;

        goto gen_arith;

    OP_32_64(xor):

        c = ARITH_XOR;

        goto gen_arith;

    case INDEX_op_shl_i32:

        c = SHIFT_SLL;

        goto gen_arith;

    case INDEX_op_shr_i32:

        c = SHIFT_SRL;

        goto gen_arith;

    case INDEX_op_sar_i32:

        c = SHIFT_SRA;

        goto gen_arith;

    case INDEX_op_mul_i32:

        c = ARITH_UMUL;

        goto gen_arith;



    OP_32_64(neg):

	c = ARITH_SUB;

	goto gen_arith1;

    OP_32_64(not):

	c = ARITH_ORN;

	goto gen_arith1;



    case INDEX_op_div_i32:

        tcg_out_div32(s, args[0], args[1], args[2], const_args[2], 0);

        break;

    case INDEX_op_divu_i32:

        tcg_out_div32(s, args[0], args[1], args[2], const_args[2], 1);

        break;



    case INDEX_op_rem_i32:

    case INDEX_op_remu_i32:

        tcg_out_div32(s, TCG_REG_I5, args[1], args[2], const_args[2],

                      opc == INDEX_op_remu_i32);

        tcg_out_arithc(s, TCG_REG_I5, TCG_REG_I5, args[2], const_args[2],

                       ARITH_UMUL);

        tcg_out_arith(s, args[0], args[1], TCG_REG_I5, ARITH_SUB);

        break;



    case INDEX_op_brcond_i32:

        tcg_out_brcond_i32(s, args[2], args[0], args[1], const_args[1],

                           args[3]);

        break;

    case INDEX_op_setcond_i32:

        tcg_out_setcond_i32(s, args[3], args[0], args[1],

                            args[2], const_args[2]);

        break;



#if TCG_TARGET_REG_BITS == 32

    case INDEX_op_brcond2_i32:

        tcg_out_brcond2_i32(s, args[4], args[0], args[1],

                            args[2], const_args[2],

                            args[3], const_args[3], args[5]);

        break;

    case INDEX_op_setcond2_i32:

        tcg_out_setcond2_i32(s, args[5], args[0], args[1], args[2],

                             args[3], const_args[3],

                             args[4], const_args[4]);

        break;

    case INDEX_op_add2_i32:

        tcg_out_arithc(s, args[0], args[2], args[4], const_args[4],

                       ARITH_ADDCC);

        tcg_out_arithc(s, args[1], args[3], args[5], const_args[5],

                       ARITH_ADDX);

        break;

    case INDEX_op_sub2_i32:

        tcg_out_arithc(s, args[0], args[2], args[4], const_args[4],

                       ARITH_SUBCC);

        tcg_out_arithc(s, args[1], args[3], args[5], const_args[5],

                       ARITH_SUBX);

        break;

    case INDEX_op_mulu2_i32:

        tcg_out_arithc(s, args[0], args[2], args[3], const_args[3],

                       ARITH_UMUL);

        tcg_out_rdy(s, args[1]);

        break;

#endif



    case INDEX_op_qemu_ld8u:

        tcg_out_qemu_ld(s, args, 0);

        break;

    case INDEX_op_qemu_ld8s:

        tcg_out_qemu_ld(s, args, 0 | 4);

        break;

    case INDEX_op_qemu_ld16u:

        tcg_out_qemu_ld(s, args, 1);

        break;

    case INDEX_op_qemu_ld16s:

        tcg_out_qemu_ld(s, args, 1 | 4);

        break;

    case INDEX_op_qemu_ld32:

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_qemu_ld32u:

#endif

        tcg_out_qemu_ld(s, args, 2);

        break;

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_qemu_ld32s:

        tcg_out_qemu_ld(s, args, 2 | 4);

        break;

#endif

    case INDEX_op_qemu_ld64:

        tcg_out_qemu_ld(s, args, 3);

        break;

    case INDEX_op_qemu_st8:

        tcg_out_qemu_st(s, args, 0);

        break;

    case INDEX_op_qemu_st16:

        tcg_out_qemu_st(s, args, 1);

        break;

    case INDEX_op_qemu_st32:

        tcg_out_qemu_st(s, args, 2);

        break;

    case INDEX_op_qemu_st64:

        tcg_out_qemu_st(s, args, 3);

        break;



#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_movi_i64:

        tcg_out_movi(s, TCG_TYPE_I64, args[0], args[1]);

        break;

    case INDEX_op_ld32s_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], LDSW);

        break;

    case INDEX_op_ld_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], LDX);

        break;

    case INDEX_op_st_i64:

        tcg_out_ldst(s, args[0], args[1], args[2], STX);

        break;

    case INDEX_op_shl_i64:

        c = SHIFT_SLLX;

        goto gen_arith;

    case INDEX_op_shr_i64:

        c = SHIFT_SRLX;

        goto gen_arith;

    case INDEX_op_sar_i64:

        c = SHIFT_SRAX;

        goto gen_arith;

    case INDEX_op_mul_i64:

        c = ARITH_MULX;

        goto gen_arith;

    case INDEX_op_div_i64:

        c = ARITH_SDIVX;

        goto gen_arith;

    case INDEX_op_divu_i64:

        c = ARITH_UDIVX;

        goto gen_arith;

    case INDEX_op_rem_i64:

    case INDEX_op_remu_i64:

        tcg_out_arithc(s, TCG_REG_I5, args[1], args[2], const_args[2],

                       opc == INDEX_op_rem_i64 ? ARITH_SDIVX : ARITH_UDIVX);

        tcg_out_arithc(s, TCG_REG_I5, TCG_REG_I5, args[2], const_args[2],

                       ARITH_MULX);

        tcg_out_arith(s, args[0], args[1], TCG_REG_I5, ARITH_SUB);

        break;

    case INDEX_op_ext32s_i64:

        if (const_args[1]) {

            tcg_out_movi(s, TCG_TYPE_I64, args[0], (int32_t)args[1]);

        } else {

            tcg_out_arithi(s, args[0], args[1], 0, SHIFT_SRA);

        }

        break;

    case INDEX_op_ext32u_i64:

        if (const_args[1]) {

            tcg_out_movi_imm32(s, args[0], args[1]);

        } else {

            tcg_out_arithi(s, args[0], args[1], 0, SHIFT_SRL);

        }

        break;



    case INDEX_op_brcond_i64:

        tcg_out_brcond_i64(s, args[2], args[0], args[1], const_args[1],

                           args[3]);

        break;

    case INDEX_op_setcond_i64:

        tcg_out_setcond_i64(s, args[3], args[0], args[1],

                            args[2], const_args[2]);

        break;



#endif

    gen_arith:

        tcg_out_arithc(s, args[0], args[1], args[2], const_args[2], c);

        break;



    gen_arith1:

	tcg_out_arithc(s, args[0], TCG_REG_G0, args[1], const_args[1], c);

	break;



    default:

        fprintf(stderr, "unknown opcode 0x%x\n", opc);

        tcg_abort();

    }

}
