static void tcg_out_qemu_ld(TCGContext *s, const TCGArg *args, bool is64)

{

    TCGReg datalo, datahi, addrlo;

    TCGReg addrhi __attribute__((unused));

    TCGMemOpIdx oi;

    TCGMemOp opc;

#if defined(CONFIG_SOFTMMU)

    int mem_index;

    TCGMemOp s_bits;

    tcg_insn_unit *label_ptr[2];

#endif



    datalo = *args++;

    datahi = (TCG_TARGET_REG_BITS == 32 && is64 ? *args++ : 0);

    addrlo = *args++;

    addrhi = (TARGET_LONG_BITS > TCG_TARGET_REG_BITS ? *args++ : 0);

    oi = *args++;

    opc = get_memop(oi);



#if defined(CONFIG_SOFTMMU)

    mem_index = get_mmuidx(oi);

    s_bits = opc & MO_SIZE;



    tcg_out_tlb_load(s, addrlo, addrhi, mem_index, s_bits,

                     label_ptr, offsetof(CPUTLBEntry, addr_read));



    /* TLB Hit.  */

    tcg_out_qemu_ld_direct(s, datalo, datahi, TCG_REG_L1, -1, 0, 0, opc);



    /* Record the current context of a load into ldst label */

    add_qemu_ldst_label(s, true, oi, datalo, datahi, addrlo, addrhi,

                        s->code_ptr, label_ptr);

#else

    {

        int32_t offset = GUEST_BASE;

        TCGReg base = addrlo;

        int index = -1;

        int seg = 0;



        /* For a 32-bit guest, the high 32 bits may contain garbage.

           We can do this with the ADDR32 prefix if we're not using

           a guest base, or when using segmentation.  Otherwise we

           need to zero-extend manually.  */

        if (GUEST_BASE == 0 || guest_base_flags) {

            seg = guest_base_flags;

            offset = 0;

            if (TCG_TARGET_REG_BITS > TARGET_LONG_BITS) {

                seg |= P_ADDR32;

            }

        } else if (TCG_TARGET_REG_BITS == 64) {

            if (TARGET_LONG_BITS == 32) {

                tcg_out_ext32u(s, TCG_REG_L0, base);

                base = TCG_REG_L0;

            }

            if (offset != GUEST_BASE) {

                tcg_out_movi(s, TCG_TYPE_I64, TCG_REG_L1, GUEST_BASE);

                index = TCG_REG_L1;

                offset = 0;

            }

        }



        tcg_out_qemu_ld_direct(s, datalo, datahi,

                               base, index, offset, seg, opc);

    }

#endif

}
