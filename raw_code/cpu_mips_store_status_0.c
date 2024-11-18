void cpu_mips_store_status(CPUMIPSState *env, target_ulong val)

{

    uint32_t mask = env->CP0_Status_rw_bitmask;

    target_ulong old = env->CP0_Status;



    if (env->insn_flags & ISA_MIPS32R6) {

        bool has_supervisor = extract32(mask, CP0St_KSU, 2) == 0x3;

#if defined(TARGET_MIPS64)

        uint32_t ksux = (1 << CP0St_KX) & val;

        ksux |= (ksux >> 1) & val; /* KX = 0 forces SX to be 0 */

        ksux |= (ksux >> 1) & val; /* SX = 0 forces UX to be 0 */

        val = (val & ~(7 << CP0St_UX)) | ksux;

#endif

        if (has_supervisor && extract32(val, CP0St_KSU, 2) == 0x3) {

            mask &= ~(3 << CP0St_KSU);

        }

        mask &= ~(((1 << CP0St_SR) | (1 << CP0St_NMI)) & val);

    }



    env->CP0_Status = (old & ~mask) | (val & mask);

#if defined(TARGET_MIPS64)

    if ((env->CP0_Status ^ old) & (old & (7 << CP0St_UX))) {

        /* Access to at least one of the 64-bit segments has been disabled */

        cpu_mips_tlb_flush(env);

    }

#endif

    if (env->CP0_Config3 & (1 << CP0C3_MT)) {

        sync_c0_status(env, env, env->current_tc);

    } else {

        compute_hflags(env);

    }

}
