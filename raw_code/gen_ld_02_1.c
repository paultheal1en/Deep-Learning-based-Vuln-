static void gen_ld(DisasContext *ctx, uint32_t opc,

                   int rt, int base, int16_t offset)

{

    TCGv t0, t1, t2;

    int mem_idx = ctx->mem_idx;



    if (rt == 0 && ctx->insn_flags & (INSN_LOONGSON2E | INSN_LOONGSON2F)) {

        /* Loongson CPU uses a load to zero register for prefetch.

           We emulate it as a NOP. On other CPU we must perform the

           actual memory access. */

        return;

    }



    t0 = tcg_temp_new();

    gen_base_offset_addr(ctx, t0, base, offset);



    switch (opc) {

#if defined(TARGET_MIPS64)

    case OPC_LWU:

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEUL |

                           ctx->default_tcg_memop_mask);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LD:

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEQ |

                           ctx->default_tcg_memop_mask);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LLD:

    case R6_OPC_LLD:

        op_ld_lld(t0, t0, mem_idx, ctx);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LDL:

        t1 = tcg_temp_new();

        /* Do a byte access to possibly trigger a page

           fault with the unaligned address.  */

        tcg_gen_qemu_ld_tl(t1, t0, mem_idx, MO_UB);

        tcg_gen_andi_tl(t1, t0, 7);

#ifndef TARGET_WORDS_BIGENDIAN

        tcg_gen_xori_tl(t1, t1, 7);

#endif

        tcg_gen_shli_tl(t1, t1, 3);

        tcg_gen_andi_tl(t0, t0, ~7);

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEQ);

        tcg_gen_shl_tl(t0, t0, t1);

        t2 = tcg_const_tl(-1);

        tcg_gen_shl_tl(t2, t2, t1);

        gen_load_gpr(t1, rt);

        tcg_gen_andc_tl(t1, t1, t2);

        tcg_temp_free(t2);

        tcg_gen_or_tl(t0, t0, t1);

        tcg_temp_free(t1);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LDR:

        t1 = tcg_temp_new();

        /* Do a byte access to possibly trigger a page

           fault with the unaligned address.  */

        tcg_gen_qemu_ld_tl(t1, t0, mem_idx, MO_UB);

        tcg_gen_andi_tl(t1, t0, 7);

#ifdef TARGET_WORDS_BIGENDIAN

        tcg_gen_xori_tl(t1, t1, 7);

#endif

        tcg_gen_shli_tl(t1, t1, 3);

        tcg_gen_andi_tl(t0, t0, ~7);

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEQ);

        tcg_gen_shr_tl(t0, t0, t1);

        tcg_gen_xori_tl(t1, t1, 63);

        t2 = tcg_const_tl(0xfffffffffffffffeull);

        tcg_gen_shl_tl(t2, t2, t1);

        gen_load_gpr(t1, rt);

        tcg_gen_and_tl(t1, t1, t2);

        tcg_temp_free(t2);

        tcg_gen_or_tl(t0, t0, t1);

        tcg_temp_free(t1);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LDPC:

        t1 = tcg_const_tl(pc_relative_pc(ctx));

        gen_op_addr_add(ctx, t0, t0, t1);

        tcg_temp_free(t1);

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEQ);

        gen_store_gpr(t0, rt);

        break;

#endif

    case OPC_LWPC:

        t1 = tcg_const_tl(pc_relative_pc(ctx));

        gen_op_addr_add(ctx, t0, t0, t1);

        tcg_temp_free(t1);

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TESL);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LWE:



    case OPC_LW:

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TESL |

                           ctx->default_tcg_memop_mask);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LHE:



    case OPC_LH:

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TESW |

                           ctx->default_tcg_memop_mask);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LHUE:



    case OPC_LHU:

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEUW |

                           ctx->default_tcg_memop_mask);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LBE:



    case OPC_LB:

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_SB);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LBUE:



    case OPC_LBU:

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_UB);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LWLE:



    case OPC_LWL:

        t1 = tcg_temp_new();

        /* Do a byte access to possibly trigger a page

           fault with the unaligned address.  */

        tcg_gen_qemu_ld_tl(t1, t0, mem_idx, MO_UB);

        tcg_gen_andi_tl(t1, t0, 3);

#ifndef TARGET_WORDS_BIGENDIAN

        tcg_gen_xori_tl(t1, t1, 3);

#endif

        tcg_gen_shli_tl(t1, t1, 3);

        tcg_gen_andi_tl(t0, t0, ~3);

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEUL);

        tcg_gen_shl_tl(t0, t0, t1);

        t2 = tcg_const_tl(-1);

        tcg_gen_shl_tl(t2, t2, t1);

        gen_load_gpr(t1, rt);

        tcg_gen_andc_tl(t1, t1, t2);

        tcg_temp_free(t2);

        tcg_gen_or_tl(t0, t0, t1);

        tcg_temp_free(t1);

        tcg_gen_ext32s_tl(t0, t0);

        gen_store_gpr(t0, rt);

        break;




    case OPC_LWR:

        t1 = tcg_temp_new();

        /* Do a byte access to possibly trigger a page

           fault with the unaligned address.  */

        tcg_gen_qemu_ld_tl(t1, t0, mem_idx, MO_UB);

        tcg_gen_andi_tl(t1, t0, 3);

#ifdef TARGET_WORDS_BIGENDIAN

        tcg_gen_xori_tl(t1, t1, 3);

#endif

        tcg_gen_shli_tl(t1, t1, 3);

        tcg_gen_andi_tl(t0, t0, ~3);

        tcg_gen_qemu_ld_tl(t0, t0, mem_idx, MO_TEUL);

        tcg_gen_shr_tl(t0, t0, t1);

        tcg_gen_xori_tl(t1, t1, 31);

        t2 = tcg_const_tl(0xfffffffeull);

        tcg_gen_shl_tl(t2, t2, t1);

        gen_load_gpr(t1, rt);

        tcg_gen_and_tl(t1, t1, t2);

        tcg_temp_free(t2);

        tcg_gen_or_tl(t0, t0, t1);

        tcg_temp_free(t1);

        tcg_gen_ext32s_tl(t0, t0);

        gen_store_gpr(t0, rt);

        break;

    case OPC_LLE:



    case OPC_LL:

    case R6_OPC_LL:

        op_ld_ll(t0, t0, mem_idx, ctx);

        gen_store_gpr(t0, rt);

        break;

    }

    tcg_temp_free(t0);

}