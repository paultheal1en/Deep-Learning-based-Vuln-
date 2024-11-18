void register_cp_regs_for_features(ARMCPU *cpu)

{

    /* Register all the coprocessor registers based on feature bits */

    CPUARMState *env = &cpu->env;

    if (arm_feature(env, ARM_FEATURE_M)) {

        /* M profile has no coprocessor registers */

        return;

    }



    define_arm_cp_regs(cpu, cp_reginfo);

    if (!arm_feature(env, ARM_FEATURE_V8)) {

        /* Must go early as it is full of wildcards that may be

         * overridden by later definitions.

         */

        define_arm_cp_regs(cpu, not_v8_cp_reginfo);

    }



    if (arm_feature(env, ARM_FEATURE_V6)) {

        /* The ID registers all have impdef reset values */

        ARMCPRegInfo v6_idregs[] = {

            { .name = "ID_PFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_pfr0 },

            { .name = "ID_PFR1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_pfr1 },

            { .name = "ID_DFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_dfr0 },

            { .name = "ID_AFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_afr0 },

            { .name = "ID_MMFR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr0 },

            { .name = "ID_MMFR1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr1 },

            { .name = "ID_MMFR2", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr2 },

            { .name = "ID_MMFR3", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 1, .opc2 = 7,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_mmfr3 },

            { .name = "ID_ISAR0", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar0 },

            { .name = "ID_ISAR1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar1 },

            { .name = "ID_ISAR2", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar2 },

            { .name = "ID_ISAR3", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar3 },

            { .name = "ID_ISAR4", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar4 },

            { .name = "ID_ISAR5", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 2, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_isar5 },

            /* 6..7 are as yet unallocated and must RAZ */

            { .name = "ID_ISAR6", .cp = 15, .crn = 0, .crm = 2,

              .opc1 = 0, .opc2 = 6, .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            { .name = "ID_ISAR7", .cp = 15, .crn = 0, .crm = 2,

              .opc1 = 0, .opc2 = 7, .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        define_arm_cp_regs(cpu, v6_idregs);

        define_arm_cp_regs(cpu, v6_cp_reginfo);

    } else {

        define_arm_cp_regs(cpu, not_v6_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_V6K)) {

        define_arm_cp_regs(cpu, v6k_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_V7)) {

        /* v7 performance monitor control register: same implementor

         * field as main ID register, and we implement only the cycle

         * count register.

         */

#ifndef CONFIG_USER_ONLY

        ARMCPRegInfo pmcr = {

            .name = "PMCR", .cp = 15, .crn = 9, .crm = 12, .opc1 = 0, .opc2 = 0,

            .access = PL0_RW, .resetvalue = cpu->midr & 0xff000000,

            .type = ARM_CP_IO,

            .fieldoffset = offsetof(CPUARMState, cp15.c9_pmcr),

            .accessfn = pmreg_access, .writefn = pmcr_write,

            .raw_writefn = raw_write,

        };

        define_one_arm_cp_reg(cpu, &pmcr);

#endif

        ARMCPRegInfo clidr = {

            .name = "CLIDR", .state = ARM_CP_STATE_BOTH,

            .opc0 = 3, .crn = 0, .crm = 0, .opc1 = 1, .opc2 = 1,

            .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = cpu->clidr

        };

        define_one_arm_cp_reg(cpu, &clidr);

        define_arm_cp_regs(cpu, v7_cp_reginfo);

    } else {

        define_arm_cp_regs(cpu, not_v7_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_V8)) {

        /* AArch64 ID registers, which all have impdef reset values */

        ARMCPRegInfo v8_idregs[] = {

            { .name = "ID_AA64PFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64pfr0 },

            { .name = "ID_AA64PFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 4, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64pfr1},

            { .name = "ID_AA64DFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              /* We mask out the PMUVer field, because we don't currently

               * implement the PMU. Not advertising it prevents the guest

               * from trying to use it and getting UNDEFs on registers we

               * don't implement.

               */

              .resetvalue = cpu->id_aa64dfr0 & ~0xf00 },

            { .name = "ID_AA64DFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64dfr1 },

            { .name = "ID_AA64AFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 4,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64afr0 },

            { .name = "ID_AA64AFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 5, .opc2 = 5,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64afr1 },

            { .name = "ID_AA64ISAR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64isar0 },

            { .name = "ID_AA64ISAR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 6, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64isar1 },

            { .name = "ID_AA64MMFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64mmfr0 },

            { .name = "ID_AA64MMFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 7, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->id_aa64mmfr1 },

            { .name = "MVFR0_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->mvfr0 },

            { .name = "MVFR1_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->mvfr1 },

            { .name = "MVFR2_EL1", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 3, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST,

              .resetvalue = cpu->mvfr2 },

            REGINFO_SENTINEL

        };

        ARMCPRegInfo rvbar = {

            .name = "RVBAR_EL1", .state = ARM_CP_STATE_AA64,

            .opc0 = 3, .opc1 = 0, .crn = 12, .crm = 0, .opc2 = 2,

            .type = ARM_CP_CONST, .access = PL1_R, .resetvalue = cpu->rvbar

        };

        define_one_arm_cp_reg(cpu, &rvbar);

        define_arm_cp_regs(cpu, v8_idregs);

        define_arm_cp_regs(cpu, v8_cp_reginfo);

        define_aarch64_debug_regs(cpu);

    }

    if (arm_feature(env, ARM_FEATURE_EL2)) {

        define_arm_cp_regs(cpu, v8_el2_cp_reginfo);

    } else {

        /* If EL2 is missing but higher ELs are enabled, we need to

         * register the no_el2 reginfos.

         */

        if (arm_feature(env, ARM_FEATURE_EL3)) {

            define_arm_cp_regs(cpu, v8_el3_no_el2_cp_reginfo);

        }

    }

    if (arm_feature(env, ARM_FEATURE_EL3)) {

        define_arm_cp_regs(cpu, v8_el3_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_MPU)) {

        /* These are the MPU registers prior to PMSAv6. Any new

         * PMSA core later than the ARM946 will require that we

         * implement the PMSAv6 or PMSAv7 registers, which are

         * completely different.

         */

        assert(!arm_feature(env, ARM_FEATURE_V6));

        define_arm_cp_regs(cpu, pmsav5_cp_reginfo);

    } else {

        define_arm_cp_regs(cpu, vmsa_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_THUMB2EE)) {

        define_arm_cp_regs(cpu, t2ee_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_GENERIC_TIMER)) {

        define_arm_cp_regs(cpu, generic_timer_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_VAPA)) {

        define_arm_cp_regs(cpu, vapa_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_CACHE_TEST_CLEAN)) {

        define_arm_cp_regs(cpu, cache_test_clean_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_CACHE_DIRTY_REG)) {

        define_arm_cp_regs(cpu, cache_dirty_status_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_CACHE_BLOCK_OPS)) {

        define_arm_cp_regs(cpu, cache_block_ops_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_OMAPCP)) {

        define_arm_cp_regs(cpu, omap_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_STRONGARM)) {

        define_arm_cp_regs(cpu, strongarm_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_XSCALE)) {

        define_arm_cp_regs(cpu, xscale_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_DUMMY_C15_REGS)) {

        define_arm_cp_regs(cpu, dummy_c15_cp_reginfo);

    }

    if (arm_feature(env, ARM_FEATURE_LPAE)) {

        define_arm_cp_regs(cpu, lpae_cp_reginfo);

    }

    /* Slightly awkwardly, the OMAP and StrongARM cores need all of

     * cp15 crn=0 to be writes-ignored, whereas for other cores they should

     * be read-only (ie write causes UNDEF exception).

     */

    {

        ARMCPRegInfo id_pre_v8_midr_cp_reginfo[] = {

            /* Pre-v8 MIDR space.

             * Note that the MIDR isn't a simple constant register because

             * of the TI925 behaviour where writes to another register can

             * cause the MIDR value to change.

             *

             * Unimplemented registers in the c15 0 0 0 space default to

             * MIDR. Define MIDR first as this entire space, then CTR, TCMTR

             * and friends override accordingly.

             */

            { .name = "MIDR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .resetvalue = cpu->midr,

              .writefn = arm_cp_write_ignore, .raw_writefn = raw_write,

              .fieldoffset = offsetof(CPUARMState, cp15.c0_cpuid),

              .type = ARM_CP_OVERRIDE },

            /* crn = 0 op1 = 0 crm = 3..7 : currently unassigned; we RAZ. */

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 3, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 4, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 5, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 6, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "DUMMY",

              .cp = 15, .crn = 0, .crm = 7, .opc1 = 0, .opc2 = CP_ANY,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        ARMCPRegInfo id_v8_midr_cp_reginfo[] = {

            /* v8 MIDR -- the wildcard isn't necessary, and nor is the

             * variable-MIDR TI925 behaviour. Instead we have a single

             * (strictly speaking IMPDEF) alias of the MIDR, REVIDR.

             */

            { .name = "MIDR_EL1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 0, .opc2 = 0,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = cpu->midr },

            { .name = "REVIDR_EL1", .state = ARM_CP_STATE_BOTH,

              .opc0 = 3, .opc1 = 0, .crn = 0, .crm = 0, .opc2 = 6,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = cpu->midr },

            REGINFO_SENTINEL

        };

        ARMCPRegInfo id_cp_reginfo[] = {

            /* These are common to v8 and pre-v8 */

            { .name = "CTR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 1,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = cpu->ctr },

            { .name = "CTR_EL0", .state = ARM_CP_STATE_AA64,

              .opc0 = 3, .opc1 = 3, .opc2 = 1, .crn = 0, .crm = 0,

              .access = PL0_R, .accessfn = ctr_el0_access,

              .type = ARM_CP_CONST, .resetvalue = cpu->ctr },

            /* TCMTR and TLBTR exist in v8 but have no 64-bit versions */

            { .name = "TCMTR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 2,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            { .name = "TLBTR",

              .cp = 15, .crn = 0, .crm = 0, .opc1 = 0, .opc2 = 3,

              .access = PL1_R, .type = ARM_CP_CONST, .resetvalue = 0 },

            REGINFO_SENTINEL

        };

        ARMCPRegInfo crn0_wi_reginfo = {

            .name = "CRN0_WI", .cp = 15, .crn = 0, .crm = CP_ANY,

            .opc1 = CP_ANY, .opc2 = CP_ANY, .access = PL1_W,

            .type = ARM_CP_NOP | ARM_CP_OVERRIDE

        };

        if (arm_feature(env, ARM_FEATURE_OMAPCP) ||

            arm_feature(env, ARM_FEATURE_STRONGARM)) {

            ARMCPRegInfo *r;

            /* Register the blanket "writes ignored" value first to cover the

             * whole space. Then update the specific ID registers to allow write

             * access, so that they ignore writes rather than causing them to

             * UNDEF.

             */

            define_one_arm_cp_reg(cpu, &crn0_wi_reginfo);

            for (r = id_pre_v8_midr_cp_reginfo;

                 r->type != ARM_CP_SENTINEL; r++) {

                r->access = PL1_RW;

            }

            for (r = id_cp_reginfo; r->type != ARM_CP_SENTINEL; r++) {

                r->access = PL1_RW;

            }

        }

        if (arm_feature(env, ARM_FEATURE_V8)) {

            define_arm_cp_regs(cpu, id_v8_midr_cp_reginfo);

        } else {

            define_arm_cp_regs(cpu, id_pre_v8_midr_cp_reginfo);

        }

        define_arm_cp_regs(cpu, id_cp_reginfo);

    }



    if (arm_feature(env, ARM_FEATURE_MPIDR)) {

        define_arm_cp_regs(cpu, mpidr_cp_reginfo);

    }



    if (arm_feature(env, ARM_FEATURE_AUXCR)) {

        ARMCPRegInfo auxcr = {

            .name = "ACTLR_EL1", .state = ARM_CP_STATE_BOTH,

            .opc0 = 3, .opc1 = 0, .crn = 1, .crm = 0, .opc2 = 1,

            .access = PL1_RW, .type = ARM_CP_CONST,

            .resetvalue = cpu->reset_auxcr

        };

        define_one_arm_cp_reg(cpu, &auxcr);

    }



    if (arm_feature(env, ARM_FEATURE_CBAR)) {

        if (arm_feature(env, ARM_FEATURE_AARCH64)) {

            /* 32 bit view is [31:18] 0...0 [43:32]. */

            uint32_t cbar32 = (extract64(cpu->reset_cbar, 18, 14) << 18)

                | extract64(cpu->reset_cbar, 32, 12);

            ARMCPRegInfo cbar_reginfo[] = {

                { .name = "CBAR",

                  .type = ARM_CP_CONST,

                  .cp = 15, .crn = 15, .crm = 0, .opc1 = 4, .opc2 = 0,

                  .access = PL1_R, .resetvalue = cpu->reset_cbar },

                { .name = "CBAR_EL1", .state = ARM_CP_STATE_AA64,

                  .type = ARM_CP_CONST,

                  .opc0 = 3, .opc1 = 1, .crn = 15, .crm = 3, .opc2 = 0,

                  .access = PL1_R, .resetvalue = cbar32 },

                REGINFO_SENTINEL

            };

            /* We don't implement a r/w 64 bit CBAR currently */

            assert(arm_feature(env, ARM_FEATURE_CBAR_RO));

            define_arm_cp_regs(cpu, cbar_reginfo);

        } else {

            ARMCPRegInfo cbar = {

                .name = "CBAR",

                .cp = 15, .crn = 15, .crm = 0, .opc1 = 4, .opc2 = 0,

                .access = PL1_R|PL3_W, .resetvalue = cpu->reset_cbar,

                .fieldoffset = offsetof(CPUARMState,

                                        cp15.c15_config_base_address)

            };

            if (arm_feature(env, ARM_FEATURE_CBAR_RO)) {

                cbar.access = PL1_R;

                cbar.fieldoffset = 0;

                cbar.type = ARM_CP_CONST;

            }

            define_one_arm_cp_reg(cpu, &cbar);

        }

    }



    /* Generic registers whose values depend on the implementation */

    {

        ARMCPRegInfo sctlr = {

            .name = "SCTLR", .state = ARM_CP_STATE_BOTH,

            .opc0 = 3, .crn = 1, .crm = 0, .opc1 = 0, .opc2 = 0,

            .access = PL1_RW, .fieldoffset = offsetof(CPUARMState, cp15.c1_sys),

            .writefn = sctlr_write, .resetvalue = cpu->reset_sctlr,

            .raw_writefn = raw_write,

        };

        if (arm_feature(env, ARM_FEATURE_XSCALE)) {

            /* Normally we would always end the TB on an SCTLR write, but Linux

             * arch/arm/mach-pxa/sleep.S expects two instructions following

             * an MMU enable to execute from cache.  Imitate this behaviour.

             */

            sctlr.type |= ARM_CP_SUPPRESS_TB_END;

        }

        define_one_arm_cp_reg(cpu, &sctlr);

    }

}
