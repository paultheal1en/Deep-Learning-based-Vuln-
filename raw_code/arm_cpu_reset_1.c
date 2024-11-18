static void arm_cpu_reset(CPUState *s)

{

    ARMCPU *cpu = ARM_CPU(s);

    ARMCPUClass *acc = ARM_CPU_GET_CLASS(cpu);

    CPUARMState *env = &cpu->env;



    acc->parent_reset(s);



    memset(env, 0, offsetof(CPUARMState, end_reset_fields));



    g_hash_table_foreach(cpu->cp_regs, cp_reg_reset, cpu);

    g_hash_table_foreach(cpu->cp_regs, cp_reg_check_reset, cpu);



    env->vfp.xregs[ARM_VFP_FPSID] = cpu->reset_fpsid;

    env->vfp.xregs[ARM_VFP_MVFR0] = cpu->mvfr0;

    env->vfp.xregs[ARM_VFP_MVFR1] = cpu->mvfr1;

    env->vfp.xregs[ARM_VFP_MVFR2] = cpu->mvfr2;



    cpu->powered_off = cpu->start_powered_off;

    s->halted = cpu->start_powered_off;



    if (arm_feature(env, ARM_FEATURE_IWMMXT)) {

        env->iwmmxt.cregs[ARM_IWMMXT_wCID] = 0x69051000 | 'Q';

    }



    if (arm_feature(env, ARM_FEATURE_AARCH64)) {

        /* 64 bit CPUs always start in 64 bit mode */

        env->aarch64 = 1;

#if defined(CONFIG_USER_ONLY)

        env->pstate = PSTATE_MODE_EL0t;

        /* Userspace expects access to DC ZVA, CTL_EL0 and the cache ops */

        env->cp15.sctlr_el[1] |= SCTLR_UCT | SCTLR_UCI | SCTLR_DZE;

        /* and to the FP/Neon instructions */

        env->cp15.cpacr_el1 = deposit64(env->cp15.cpacr_el1, 20, 2, 3);

#else

        /* Reset into the highest available EL */

        if (arm_feature(env, ARM_FEATURE_EL3)) {

            env->pstate = PSTATE_MODE_EL3h;

        } else if (arm_feature(env, ARM_FEATURE_EL2)) {

            env->pstate = PSTATE_MODE_EL2h;

        } else {

            env->pstate = PSTATE_MODE_EL1h;

        }

        env->pc = cpu->rvbar;

#endif

    } else {

#if defined(CONFIG_USER_ONLY)

        /* Userspace expects access to cp10 and cp11 for FP/Neon */

        env->cp15.cpacr_el1 = deposit64(env->cp15.cpacr_el1, 20, 4, 0xf);

#endif

    }



#if defined(CONFIG_USER_ONLY)

    env->uncached_cpsr = ARM_CPU_MODE_USR;

    /* For user mode we must enable access to coprocessors */

    env->vfp.xregs[ARM_VFP_FPEXC] = 1 << 30;

    if (arm_feature(env, ARM_FEATURE_IWMMXT)) {

        env->cp15.c15_cpar = 3;

    } else if (arm_feature(env, ARM_FEATURE_XSCALE)) {

        env->cp15.c15_cpar = 1;

    }

#else

    /* SVC mode with interrupts disabled.  */

    env->uncached_cpsr = ARM_CPU_MODE_SVC;

    env->daif = PSTATE_D | PSTATE_A | PSTATE_I | PSTATE_F;



    if (arm_feature(env, ARM_FEATURE_M)) {

        uint32_t initial_msp; /* Loaded from 0x0 */

        uint32_t initial_pc; /* Loaded from 0x4 */

        uint8_t *rom;



        /* For M profile we store FAULTMASK and PRIMASK in the

         * PSTATE F and I bits; these are both clear at reset.

         */

        env->daif &= ~(PSTATE_I | PSTATE_F);



        /* The reset value of this bit is IMPDEF, but ARM recommends

         * that it resets to 1, so QEMU always does that rather than making

         * it dependent on CPU model.

         */

        env->v7m.ccr = R_V7M_CCR_STKALIGN_MASK;



        /* Unlike A/R profile, M profile defines the reset LR value */

        env->regs[14] = 0xffffffff;



        /* Load the initial SP and PC from the vector table at address 0 */

        rom = rom_ptr(0);

        if (rom) {

            /* Address zero is covered by ROM which hasn't yet been

             * copied into physical memory.

             */

            initial_msp = ldl_p(rom);

            initial_pc = ldl_p(rom + 4);

        } else {

            /* Address zero not covered by a ROM blob, or the ROM blob

             * is in non-modifiable memory and this is a second reset after

             * it got copied into memory. In the latter case, rom_ptr

             * will return a NULL pointer and we should use ldl_phys instead.

             */

            initial_msp = ldl_phys(s->as, 0);

            initial_pc = ldl_phys(s->as, 4);

        }



        env->regs[13] = initial_msp & 0xFFFFFFFC;

        env->regs[15] = initial_pc & ~1;

        env->thumb = initial_pc & 1;

    }



    /* AArch32 has a hard highvec setting of 0xFFFF0000.  If we are currently

     * executing as AArch32 then check if highvecs are enabled and

     * adjust the PC accordingly.

     */

    if (A32_BANKED_CURRENT_REG_GET(env, sctlr) & SCTLR_V) {

        env->regs[15] = 0xFFFF0000;

    }



    env->vfp.xregs[ARM_VFP_FPEXC] = 0;

#endif

    set_flush_to_zero(1, &env->vfp.standard_fp_status);

    set_flush_inputs_to_zero(1, &env->vfp.standard_fp_status);

    set_default_nan_mode(1, &env->vfp.standard_fp_status);

    set_float_detect_tininess(float_tininess_before_rounding,

                              &env->vfp.fp_status);

    set_float_detect_tininess(float_tininess_before_rounding,

                              &env->vfp.standard_fp_status);

#ifndef CONFIG_USER_ONLY

    if (kvm_enabled()) {

        kvm_arm_reset_vcpu(cpu);

    }

#endif



    hw_breakpoint_update_all(cpu);

    hw_watchpoint_update_all(cpu);

}
