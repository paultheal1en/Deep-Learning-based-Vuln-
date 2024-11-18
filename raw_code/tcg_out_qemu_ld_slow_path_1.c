static void tcg_out_qemu_ld_slow_path(TCGContext *s, TCGLabelQemuLdst *l)

{

    TCGMemOp opc = l->opc;

    TCGReg data_reg;

    uint8_t **label_ptr = &l->label_ptr[0];



    /* resolve label address */

    *(uint32_t *)label_ptr[0] = (uint32_t)(s->code_ptr - label_ptr[0] - 4);

    if (TARGET_LONG_BITS > TCG_TARGET_REG_BITS) {

        *(uint32_t *)label_ptr[1] = (uint32_t)(s->code_ptr - label_ptr[1] - 4);

    }



    if (TCG_TARGET_REG_BITS == 32) {

        int ofs = 0;



        tcg_out_st(s, TCG_TYPE_PTR, TCG_AREG0, TCG_REG_ESP, ofs);

        ofs += 4;



        tcg_out_st(s, TCG_TYPE_I32, l->addrlo_reg, TCG_REG_ESP, ofs);

        ofs += 4;



        if (TARGET_LONG_BITS == 64) {

            tcg_out_st(s, TCG_TYPE_I32, l->addrhi_reg, TCG_REG_ESP, ofs);

            ofs += 4;

        }



        tcg_out_sti(s, TCG_TYPE_I32, TCG_REG_ESP, ofs, l->mem_index);

        ofs += 4;



        tcg_out_sti(s, TCG_TYPE_I32, TCG_REG_ESP, ofs, (uintptr_t)l->raddr);

    } else {

        tcg_out_mov(s, TCG_TYPE_PTR, tcg_target_call_iarg_regs[0], TCG_AREG0);

        /* The second argument is already loaded with addrlo.  */

        tcg_out_movi(s, TCG_TYPE_I32, tcg_target_call_iarg_regs[2],

                     l->mem_index);

        tcg_out_movi(s, TCG_TYPE_PTR, tcg_target_call_iarg_regs[3],

                     (uintptr_t)l->raddr);

    }



    tcg_out_calli(s, (uintptr_t)qemu_ld_helpers[opc & ~MO_SIGN]);



    data_reg = l->datalo_reg;

    switch (opc & MO_SSIZE) {

    case MO_SB:

        tcg_out_ext8s(s, data_reg, TCG_REG_EAX, P_REXW);

        break;

    case MO_SW:

        tcg_out_ext16s(s, data_reg, TCG_REG_EAX, P_REXW);

        break;

#if TCG_TARGET_REG_BITS == 64

    case MO_SL:

        tcg_out_ext32s(s, data_reg, TCG_REG_EAX);

        break;

#endif

    case MO_UB:

    case MO_UW:

        /* Note that the helpers have zero-extended to tcg_target_long.  */

    case MO_UL:

        tcg_out_mov(s, TCG_TYPE_I32, data_reg, TCG_REG_EAX);

        break;

    case MO_Q:

        if (TCG_TARGET_REG_BITS == 64) {

            tcg_out_mov(s, TCG_TYPE_I64, data_reg, TCG_REG_RAX);

        } else if (data_reg == TCG_REG_EDX) {

            /* xchg %edx, %eax */

            tcg_out_opc(s, OPC_XCHG_ax_r32 + TCG_REG_EDX, 0, 0, 0);

            tcg_out_mov(s, TCG_TYPE_I32, l->datahi_reg, TCG_REG_EAX);

        } else {

            tcg_out_mov(s, TCG_TYPE_I32, data_reg, TCG_REG_EAX);

            tcg_out_mov(s, TCG_TYPE_I32, l->datahi_reg, TCG_REG_EDX);

        }

        break;

    default:

        tcg_abort();

    }



    /* Jump to the code corresponding to next IR of qemu_st */

    tcg_out_jmp(s, (uintptr_t)l->raddr);

}
