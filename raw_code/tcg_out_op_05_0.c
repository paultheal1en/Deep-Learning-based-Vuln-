static void tcg_out_op(TCGContext *s, TCGOpcode opc, const TCGArg *args,

                       const int *const_args)

{

    uint8_t *old_code_ptr = s->code_ptr;



    tcg_out_op_t(s, opc);



    switch (opc) {

    case INDEX_op_exit_tb:

        tcg_out64(s, args[0]);

        break;

    case INDEX_op_goto_tb:

        if (s->tb_jmp_offset) {

            /* Direct jump method. */

            assert(args[0] < ARRAY_SIZE(s->tb_jmp_offset));

            s->tb_jmp_offset[args[0]] = s->code_ptr - s->code_buf;

            tcg_out32(s, 0);

        } else {

            /* Indirect jump method. */

            TODO();

        }

        assert(args[0] < ARRAY_SIZE(s->tb_next_offset));

        s->tb_next_offset[args[0]] = s->code_ptr - s->code_buf;

        break;

    case INDEX_op_br:

        tci_out_label(s, args[0]);

        break;

    case INDEX_op_call:

        tcg_out_ri(s, const_args[0], args[0]);

        break;

    case INDEX_op_setcond_i32:

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_ri32(s, const_args[2], args[2]);

        tcg_out8(s, args[3]);   /* condition */

        break;

#if TCG_TARGET_REG_BITS == 32

    case INDEX_op_setcond2_i32:

        /* setcond2_i32 cond, t0, t1_low, t1_high, t2_low, t2_high */

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_r(s, args[2]);

        tcg_out_ri32(s, const_args[3], args[3]);

        tcg_out_ri32(s, const_args[4], args[4]);

        tcg_out8(s, args[5]);   /* condition */

        break;

#elif TCG_TARGET_REG_BITS == 64

    case INDEX_op_setcond_i64:

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_ri64(s, const_args[2], args[2]);

        tcg_out8(s, args[3]);   /* condition */

        break;

#endif

    case INDEX_op_movi_i32:

        TODO(); /* Handled by tcg_out_movi? */

        break;

    case INDEX_op_ld8u_i32:

    case INDEX_op_ld8s_i32:

    case INDEX_op_ld16u_i32:

    case INDEX_op_ld16s_i32:

    case INDEX_op_ld_i32:

    case INDEX_op_st8_i32:

    case INDEX_op_st16_i32:

    case INDEX_op_st_i32:

    case INDEX_op_ld8u_i64:

    case INDEX_op_ld8s_i64:

    case INDEX_op_ld16u_i64:

    case INDEX_op_ld16s_i64:

    case INDEX_op_ld32u_i64:

    case INDEX_op_ld32s_i64:

    case INDEX_op_ld_i64:

    case INDEX_op_st8_i64:

    case INDEX_op_st16_i64:

    case INDEX_op_st32_i64:

    case INDEX_op_st_i64:

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        assert(args[2] == (uint32_t)args[2]);

        tcg_out32(s, args[2]);

        break;

    case INDEX_op_add_i32:

    case INDEX_op_sub_i32:

    case INDEX_op_mul_i32:

    case INDEX_op_and_i32:

    case INDEX_op_andc_i32:     /* Optional (TCG_TARGET_HAS_andc_i32). */

    case INDEX_op_eqv_i32:      /* Optional (TCG_TARGET_HAS_eqv_i32). */

    case INDEX_op_nand_i32:     /* Optional (TCG_TARGET_HAS_nand_i32). */

    case INDEX_op_nor_i32:      /* Optional (TCG_TARGET_HAS_nor_i32). */

    case INDEX_op_or_i32:

    case INDEX_op_orc_i32:      /* Optional (TCG_TARGET_HAS_orc_i32). */

    case INDEX_op_xor_i32:

    case INDEX_op_shl_i32:

    case INDEX_op_shr_i32:

    case INDEX_op_sar_i32:

    case INDEX_op_rotl_i32:     /* Optional (TCG_TARGET_HAS_rot_i32). */

    case INDEX_op_rotr_i32:     /* Optional (TCG_TARGET_HAS_rot_i32). */

        tcg_out_r(s, args[0]);

        tcg_out_ri32(s, const_args[1], args[1]);

        tcg_out_ri32(s, const_args[2], args[2]);

        break;

    case INDEX_op_deposit_i32:  /* Optional (TCG_TARGET_HAS_deposit_i32). */

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_r(s, args[2]);

        assert(args[3] <= UINT8_MAX);

        tcg_out8(s, args[3]);

        assert(args[4] <= UINT8_MAX);

        tcg_out8(s, args[4]);

        break;



#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_mov_i64:

    case INDEX_op_movi_i64:

        TODO();

        break;

    case INDEX_op_add_i64:

    case INDEX_op_sub_i64:

    case INDEX_op_mul_i64:

    case INDEX_op_and_i64:

    case INDEX_op_andc_i64:     /* Optional (TCG_TARGET_HAS_andc_i64). */

    case INDEX_op_eqv_i64:      /* Optional (TCG_TARGET_HAS_eqv_i64). */

    case INDEX_op_nand_i64:     /* Optional (TCG_TARGET_HAS_nand_i64). */

    case INDEX_op_nor_i64:      /* Optional (TCG_TARGET_HAS_nor_i64). */

    case INDEX_op_or_i64:

    case INDEX_op_orc_i64:      /* Optional (TCG_TARGET_HAS_orc_i64). */

    case INDEX_op_xor_i64:

    case INDEX_op_shl_i64:

    case INDEX_op_shr_i64:

    case INDEX_op_sar_i64:

        /* TODO: Implementation of rotl_i64, rotr_i64 missing in tci.c. */

    case INDEX_op_rotl_i64:     /* Optional (TCG_TARGET_HAS_rot_i64). */

    case INDEX_op_rotr_i64:     /* Optional (TCG_TARGET_HAS_rot_i64). */

        tcg_out_r(s, args[0]);

        tcg_out_ri64(s, const_args[1], args[1]);

        tcg_out_ri64(s, const_args[2], args[2]);

        break;

    case INDEX_op_deposit_i64:  /* Optional (TCG_TARGET_HAS_deposit_i64). */

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_r(s, args[2]);

        assert(args[3] <= UINT8_MAX);

        tcg_out8(s, args[3]);

        assert(args[4] <= UINT8_MAX);

        tcg_out8(s, args[4]);

        break;

    case INDEX_op_div_i64:      /* Optional (TCG_TARGET_HAS_div_i64). */

    case INDEX_op_divu_i64:     /* Optional (TCG_TARGET_HAS_div_i64). */

    case INDEX_op_rem_i64:      /* Optional (TCG_TARGET_HAS_div_i64). */

    case INDEX_op_remu_i64:     /* Optional (TCG_TARGET_HAS_div_i64). */

        TODO();

        break;

    case INDEX_op_div2_i64:     /* Optional (TCG_TARGET_HAS_div2_i64). */

    case INDEX_op_divu2_i64:    /* Optional (TCG_TARGET_HAS_div2_i64). */

        TODO();

        break;

    case INDEX_op_brcond_i64:

        tcg_out_r(s, args[0]);

        tcg_out_ri64(s, const_args[1], args[1]);

        tcg_out8(s, args[2]);           /* condition */

        tci_out_label(s, args[3]);

        break;

    case INDEX_op_bswap16_i64:  /* Optional (TCG_TARGET_HAS_bswap16_i64). */

    case INDEX_op_bswap32_i64:  /* Optional (TCG_TARGET_HAS_bswap32_i64). */

    case INDEX_op_bswap64_i64:  /* Optional (TCG_TARGET_HAS_bswap64_i64). */

    case INDEX_op_not_i64:      /* Optional (TCG_TARGET_HAS_not_i64). */

    case INDEX_op_neg_i64:      /* Optional (TCG_TARGET_HAS_neg_i64). */

    case INDEX_op_ext8s_i64:    /* Optional (TCG_TARGET_HAS_ext8s_i64). */

    case INDEX_op_ext8u_i64:    /* Optional (TCG_TARGET_HAS_ext8u_i64). */

    case INDEX_op_ext16s_i64:   /* Optional (TCG_TARGET_HAS_ext16s_i64). */

    case INDEX_op_ext16u_i64:   /* Optional (TCG_TARGET_HAS_ext16u_i64). */

    case INDEX_op_ext32s_i64:   /* Optional (TCG_TARGET_HAS_ext32s_i64). */

    case INDEX_op_ext32u_i64:   /* Optional (TCG_TARGET_HAS_ext32u_i64). */

#endif /* TCG_TARGET_REG_BITS == 64 */

    case INDEX_op_neg_i32:      /* Optional (TCG_TARGET_HAS_neg_i32). */

    case INDEX_op_not_i32:      /* Optional (TCG_TARGET_HAS_not_i32). */

    case INDEX_op_ext8s_i32:    /* Optional (TCG_TARGET_HAS_ext8s_i32). */

    case INDEX_op_ext16s_i32:   /* Optional (TCG_TARGET_HAS_ext16s_i32). */

    case INDEX_op_ext8u_i32:    /* Optional (TCG_TARGET_HAS_ext8u_i32). */

    case INDEX_op_ext16u_i32:   /* Optional (TCG_TARGET_HAS_ext16u_i32). */

    case INDEX_op_bswap16_i32:  /* Optional (TCG_TARGET_HAS_bswap16_i32). */

    case INDEX_op_bswap32_i32:  /* Optional (TCG_TARGET_HAS_bswap32_i32). */

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        break;

    case INDEX_op_div_i32:      /* Optional (TCG_TARGET_HAS_div_i32). */

    case INDEX_op_divu_i32:     /* Optional (TCG_TARGET_HAS_div_i32). */

    case INDEX_op_rem_i32:      /* Optional (TCG_TARGET_HAS_div_i32). */

    case INDEX_op_remu_i32:     /* Optional (TCG_TARGET_HAS_div_i32). */

        tcg_out_r(s, args[0]);

        tcg_out_ri32(s, const_args[1], args[1]);

        tcg_out_ri32(s, const_args[2], args[2]);

        break;

    case INDEX_op_div2_i32:     /* Optional (TCG_TARGET_HAS_div2_i32). */

    case INDEX_op_divu2_i32:    /* Optional (TCG_TARGET_HAS_div2_i32). */

        TODO();

        break;

#if TCG_TARGET_REG_BITS == 32

    case INDEX_op_add2_i32:

    case INDEX_op_sub2_i32:

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_r(s, args[2]);

        tcg_out_r(s, args[3]);

        tcg_out_r(s, args[4]);

        tcg_out_r(s, args[5]);

        break;

    case INDEX_op_brcond2_i32:

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_ri32(s, const_args[2], args[2]);

        tcg_out_ri32(s, const_args[3], args[3]);

        tcg_out8(s, args[4]);           /* condition */

        tci_out_label(s, args[5]);

        break;

    case INDEX_op_mulu2_i32:

        tcg_out_r(s, args[0]);

        tcg_out_r(s, args[1]);

        tcg_out_r(s, args[2]);

        tcg_out_r(s, args[3]);

        break;

#endif

    case INDEX_op_brcond_i32:

        tcg_out_r(s, args[0]);

        tcg_out_ri32(s, const_args[1], args[1]);

        tcg_out8(s, args[2]);           /* condition */

        tci_out_label(s, args[3]);

        break;

    case INDEX_op_qemu_ld8u:

    case INDEX_op_qemu_ld8s:

    case INDEX_op_qemu_ld16u:

    case INDEX_op_qemu_ld16s:

    case INDEX_op_qemu_ld32:

#if TCG_TARGET_REG_BITS == 64

    case INDEX_op_qemu_ld32s:

    case INDEX_op_qemu_ld32u:

#endif

        tcg_out_r(s, *args++);

        tcg_out_r(s, *args++);

#if TARGET_LONG_BITS > TCG_TARGET_REG_BITS

        tcg_out_r(s, *args++);

#endif

#ifdef CONFIG_SOFTMMU

        tcg_out_i(s, *args);

#endif

        break;

    case INDEX_op_qemu_ld64:

        tcg_out_r(s, *args++);

#if TCG_TARGET_REG_BITS == 32

        tcg_out_r(s, *args++);

#endif

        tcg_out_r(s, *args++);

#if TARGET_LONG_BITS > TCG_TARGET_REG_BITS

        tcg_out_r(s, *args++);

#endif

#ifdef CONFIG_SOFTMMU

        tcg_out_i(s, *args);

#endif

        break;

    case INDEX_op_qemu_st8:

    case INDEX_op_qemu_st16:

    case INDEX_op_qemu_st32:

        tcg_out_r(s, *args++);

        tcg_out_r(s, *args++);

#if TARGET_LONG_BITS > TCG_TARGET_REG_BITS

        tcg_out_r(s, *args++);

#endif

#ifdef CONFIG_SOFTMMU

        tcg_out_i(s, *args);

#endif

        break;

    case INDEX_op_qemu_st64:

        tcg_out_r(s, *args++);

#if TCG_TARGET_REG_BITS == 32

        tcg_out_r(s, *args++);

#endif

        tcg_out_r(s, *args++);

#if TARGET_LONG_BITS > TCG_TARGET_REG_BITS

        tcg_out_r(s, *args++);

#endif

#ifdef CONFIG_SOFTMMU

        tcg_out_i(s, *args);

#endif

        break;

    case INDEX_op_end:

        TODO();

        break;

    default:

        fprintf(stderr, "Missing: %s\n", tcg_op_defs[opc].name);

        tcg_abort();

    }

    old_code_ptr[1] = s->code_ptr - old_code_ptr;

}
