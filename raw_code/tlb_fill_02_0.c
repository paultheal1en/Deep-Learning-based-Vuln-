void tlb_fill(CPUState *cs, target_ulong addr, int is_write, int mmu_idx,

              uintptr_t retaddr)

{

    bool ret;

    uint32_t fsr = 0;

    ARMMMUFaultInfo fi = {};



    ret = arm_tlb_fill(cs, addr, is_write, mmu_idx, &fsr, &fi);

    if (unlikely(ret)) {

        ARMCPU *cpu = ARM_CPU(cs);

        CPUARMState *env = &cpu->env;

        uint32_t syn, exc;

        unsigned int target_el;

        bool same_el;



        if (retaddr) {

            /* now we have a real cpu fault */

            cpu_restore_state(cs, retaddr);

        }



        target_el = exception_target_el(env);

        if (fi.stage2) {

            target_el = 2;

            env->cp15.hpfar_el2 = extract64(fi.s2addr, 12, 47) << 4;

        }

        same_el = arm_current_el(env) == target_el;

        /* AArch64 syndrome does not have an LPAE bit */

        syn = fsr & ~(1 << 9);



        /* For insn and data aborts we assume there is no instruction syndrome

         * information; this is always true for exceptions reported to EL1.

         */

        if (is_write == 2) {

            syn = syn_insn_abort(same_el, 0, fi.s1ptw, syn);

            exc = EXCP_PREFETCH_ABORT;

        } else {

            syn = syn_data_abort(same_el, 0, 0, fi.s1ptw, is_write == 1, syn);

            if (is_write == 1 && arm_feature(env, ARM_FEATURE_V6)) {

                fsr |= (1 << 11);

            }

            exc = EXCP_DATA_ABORT;

        }



        env->exception.vaddress = addr;

        env->exception.fsr = fsr;

        raise_exception(env, exc, syn, target_el);

    }

}
