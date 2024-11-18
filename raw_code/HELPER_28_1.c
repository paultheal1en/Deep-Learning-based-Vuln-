void HELPER(set_cp15)(CPUState *env, uint32_t insn, uint32_t val)

{

    int op1;

    int op2;

    int crm;



    op1 = (insn >> 21) & 7;

    op2 = (insn >> 5) & 7;

    crm = insn & 0xf;

    switch ((insn >> 16) & 0xf) {

    case 0:

        /* ID codes.  */

        if (arm_feature(env, ARM_FEATURE_XSCALE))

            break;

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            break;

        if (arm_feature(env, ARM_FEATURE_V7)

                && op1 == 2 && crm == 0 && op2 == 0) {

            env->cp15.c0_cssel = val & 0xf;

            break;

        }

        goto bad_reg;

    case 1: /* System configuration.  */

        if (arm_feature(env, ARM_FEATURE_V7)

                && op1 == 0 && crm == 1 && op2 == 0) {

            env->cp15.c1_scr = val;

            break;

        }

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            op2 = 0;

        switch (op2) {

        case 0:

            if (!arm_feature(env, ARM_FEATURE_XSCALE) || crm == 0)

                env->cp15.c1_sys = val;

            /* ??? Lots of these bits are not implemented.  */

            /* This may enable/disable the MMU, so do a TLB flush.  */

            tlb_flush(env, 1);

            break;

        case 1: /* Auxiliary control register.  */

            if (arm_feature(env, ARM_FEATURE_XSCALE)) {

                env->cp15.c1_xscaleauxcr = val;

                break;

            }

            /* Not implemented.  */

            break;

        case 2:

            if (arm_feature(env, ARM_FEATURE_XSCALE))

                goto bad_reg;

            if (env->cp15.c1_coproc != val) {

                env->cp15.c1_coproc = val;

                /* ??? Is this safe when called from within a TB?  */

                tb_flush(env);

            }

            break;

        default:

            goto bad_reg;

        }

        break;

    case 2: /* MMU Page table control / MPU cache control.  */

        if (arm_feature(env, ARM_FEATURE_MPU)) {

            switch (op2) {

            case 0:

                env->cp15.c2_data = val;

                break;

            case 1:

                env->cp15.c2_insn = val;

                break;

            default:

                goto bad_reg;

            }

        } else {

	    switch (op2) {

	    case 0:

		env->cp15.c2_base0 = val;

		break;

	    case 1:

		env->cp15.c2_base1 = val;

		break;

	    case 2:

                val &= 7;

                env->cp15.c2_control = val;

		env->cp15.c2_mask = ~(((uint32_t)0xffffffffu) >> val);

                env->cp15.c2_base_mask = ~((uint32_t)0x3fffu >> val);

		break;

	    default:

		goto bad_reg;

	    }

        }

        break;

    case 3: /* MMU Domain access control / MPU write buffer control.  */

        env->cp15.c3 = val;

        tlb_flush(env, 1); /* Flush TLB as domain not tracked in TLB */

        break;

    case 4: /* Reserved.  */

        goto bad_reg;

    case 5: /* MMU Fault status / MPU access permission.  */

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            op2 = 0;

        switch (op2) {

        case 0:

            if (arm_feature(env, ARM_FEATURE_MPU))

                val = extended_mpu_ap_bits(val);

            env->cp15.c5_data = val;

            break;

        case 1:

            if (arm_feature(env, ARM_FEATURE_MPU))

                val = extended_mpu_ap_bits(val);

            env->cp15.c5_insn = val;

            break;

        case 2:

            if (!arm_feature(env, ARM_FEATURE_MPU))

                goto bad_reg;

            env->cp15.c5_data = val;

            break;

        case 3:

            if (!arm_feature(env, ARM_FEATURE_MPU))

                goto bad_reg;

            env->cp15.c5_insn = val;

            break;

        default:

            goto bad_reg;

        }

        break;

    case 6: /* MMU Fault address / MPU base/size.  */

        if (arm_feature(env, ARM_FEATURE_MPU)) {

            if (crm >= 8)

                goto bad_reg;

            env->cp15.c6_region[crm] = val;

        } else {

            if (arm_feature(env, ARM_FEATURE_OMAPCP))

                op2 = 0;

            switch (op2) {

            case 0:

                env->cp15.c6_data = val;

                break;

            case 1: /* ??? This is WFAR on armv6 */

            case 2:

                env->cp15.c6_insn = val;

                break;

            default:

                goto bad_reg;

            }

        }

        break;

    case 7: /* Cache control.  */

        env->cp15.c15_i_max = 0x000;

        env->cp15.c15_i_min = 0xff0;

        if (op1 != 0) {

            goto bad_reg;

        }

        /* No cache, so nothing to do except VA->PA translations. */

        if (arm_feature(env, ARM_FEATURE_VAPA)) {

            switch (crm) {

            case 4:

                if (arm_feature(env, ARM_FEATURE_V7)) {

                    env->cp15.c7_par = val & 0xfffff6ff;

                } else {

                    env->cp15.c7_par = val & 0xfffff1ff;

                }

                break;

            case 8: {

                uint32_t phys_addr;

                target_ulong page_size;

                int prot;

                int ret, is_user = op2 & 2;

                int access_type = op2 & 1;



                if (op2 & 4) {

                    /* Other states are only available with TrustZone */

                    goto bad_reg;

                }

                ret = get_phys_addr(env, val, access_type, is_user,

                                    &phys_addr, &prot, &page_size);

                if (ret == 0) {

                    /* We do not set any attribute bits in the PAR */

                    if (page_size == (1 << 24)

                        && arm_feature(env, ARM_FEATURE_V7)) {

                        env->cp15.c7_par = (phys_addr & 0xff000000) | 1 << 1;

                    } else {

                        env->cp15.c7_par = phys_addr & 0xfffff000;

                    }

                } else {

                    env->cp15.c7_par = ((ret & (10 << 1)) >> 5) |

                                       ((ret & (12 << 1)) >> 6) |

                                       ((ret & 0xf) << 1) | 1;

                }

                break;

            }

            }

        }

        break;

    case 8: /* MMU TLB control.  */

        switch (op2) {

        case 0: /* Invalidate all.  */

            tlb_flush(env, 0);

            break;

        case 1: /* Invalidate single TLB entry.  */

            tlb_flush_page(env, val & TARGET_PAGE_MASK);

            break;

        case 2: /* Invalidate on ASID.  */

            tlb_flush(env, val == 0);

            break;

        case 3: /* Invalidate single entry on MVA.  */

            /* ??? This is like case 1, but ignores ASID.  */

            tlb_flush(env, 1);

            break;

        default:

            goto bad_reg;

        }

        break;

    case 9:

        if (arm_feature(env, ARM_FEATURE_OMAPCP))

            break;

        if (arm_feature(env, ARM_FEATURE_STRONGARM))

            break; /* Ignore ReadBuffer access */

        switch (crm) {

        case 0: /* Cache lockdown.  */

	    switch (op1) {

	    case 0: /* L1 cache.  */

		switch (op2) {

		case 0:

		    env->cp15.c9_data = val;

		    break;

		case 1:

		    env->cp15.c9_insn = val;

		    break;

		default:

		    goto bad_reg;

		}

		break;

	    case 1: /* L2 cache.  */

		/* Ignore writes to L2 lockdown/auxiliary registers.  */

		break;

	    default:

		goto bad_reg;

	    }

	    break;

        case 1: /* TCM memory region registers.  */

            /* Not implemented.  */

            goto bad_reg;

        case 12: /* Performance monitor control */

            /* Performance monitors are implementation defined in v7,

             * but with an ARM recommended set of registers, which we

             * follow (although we don't actually implement any counters)

             */

            if (!arm_feature(env, ARM_FEATURE_V7)) {

                goto bad_reg;

            }

            switch (op2) {

            case 0: /* performance monitor control register */

                /* only the DP, X, D and E bits are writable */

                env->cp15.c9_pmcr &= ~0x39;

                env->cp15.c9_pmcr |= (val & 0x39);

                break;

            case 1: /* Count enable set register */

                val &= (1 << 31);

                env->cp15.c9_pmcnten |= val;

                break;

            case 2: /* Count enable clear */

                val &= (1 << 31);

                env->cp15.c9_pmcnten &= ~val;

                break;

            case 3: /* Overflow flag status */

                env->cp15.c9_pmovsr &= ~val;

                break;

            case 4: /* Software increment */

                /* RAZ/WI since we don't implement the software-count event */

                break;

            case 5: /* Event counter selection register */

                /* Since we don't implement any events, writing to this register

                 * is actually UNPREDICTABLE. So we choose to RAZ/WI.

                 */

                break;

            default:

                goto bad_reg;

            }

            break;

        case 13: /* Performance counters */

            if (!arm_feature(env, ARM_FEATURE_V7)) {

                goto bad_reg;

            }

            switch (op2) {

            case 0: /* Cycle count register: not implemented, so RAZ/WI */

                break;

            case 1: /* Event type select */

                env->cp15.c9_pmxevtyper = val & 0xff;

                break;

            case 2: /* Event count register */

                /* Unimplemented (we have no events), RAZ/WI */

                break;

            default:

                goto bad_reg;

            }

            break;

        case 14: /* Performance monitor control */

            if (!arm_feature(env, ARM_FEATURE_V7)) {

                goto bad_reg;

            }

            switch (op2) {

            case 0: /* user enable */

                env->cp15.c9_pmuserenr = val & 1;

                /* changes access rights for cp registers, so flush tbs */

                tb_flush(env);

                break;

            case 1: /* interrupt enable set */

                /* We have no event counters so only the C bit can be changed */

                val &= (1 << 31);

                env->cp15.c9_pminten |= val;

                break;

            case 2: /* interrupt enable clear */

                val &= (1 << 31);

                env->cp15.c9_pminten &= ~val;

                break;

            }

            break;

        default:

            goto bad_reg;

        }

        break;

    case 10: /* MMU TLB lockdown.  */

        /* ??? TLB lockdown not implemented.  */

        break;

    case 12: /* Reserved.  */

        goto bad_reg;

    case 13: /* Process ID.  */

        switch (op2) {

        case 0:

            /* Unlike real hardware the qemu TLB uses virtual addresses,

               not modified virtual addresses, so this causes a TLB flush.

             */

            if (env->cp15.c13_fcse != val)

              tlb_flush(env, 1);

            env->cp15.c13_fcse = val;

            break;

        case 1:

            /* This changes the ASID, so do a TLB flush.  */

            if (env->cp15.c13_context != val

                && !arm_feature(env, ARM_FEATURE_MPU))

              tlb_flush(env, 0);

            env->cp15.c13_context = val;

            break;

        default:

            goto bad_reg;

        }

        break;

    case 14: /* Reserved.  */

        goto bad_reg;

    case 15: /* Implementation specific.  */

        if (arm_feature(env, ARM_FEATURE_XSCALE)) {

            if (op2 == 0 && crm == 1) {

                if (env->cp15.c15_cpar != (val & 0x3fff)) {

                    /* Changes cp0 to cp13 behavior, so needs a TB flush.  */

                    tb_flush(env);

                    env->cp15.c15_cpar = val & 0x3fff;

                }

                break;

            }

            goto bad_reg;

        }

        if (arm_feature(env, ARM_FEATURE_OMAPCP)) {

            switch (crm) {

            case 0:

                break;

            case 1: /* Set TI925T configuration.  */

                env->cp15.c15_ticonfig = val & 0xe7;

                env->cp15.c0_cpuid = (val & (1 << 5)) ? /* OS_TYPE bit */

                        ARM_CPUID_TI915T : ARM_CPUID_TI925T;

                break;

            case 2: /* Set I_max.  */

                env->cp15.c15_i_max = val;

                break;

            case 3: /* Set I_min.  */

                env->cp15.c15_i_min = val;

                break;

            case 4: /* Set thread-ID.  */

                env->cp15.c15_threadid = val & 0xffff;

                break;

            case 8: /* Wait-for-interrupt (deprecated).  */

                cpu_interrupt(env, CPU_INTERRUPT_HALT);

                break;

            default:

                goto bad_reg;

            }

        }

        if (ARM_CPUID(env) == ARM_CPUID_CORTEXA9) {

            switch (crm) {

            case 0:

                if ((op1 == 0) && (op2 == 0)) {

                    env->cp15.c15_power_control = val;

                } else if ((op1 == 0) && (op2 == 1)) {

                    env->cp15.c15_diagnostic = val;

                } else if ((op1 == 0) && (op2 == 2)) {

                    env->cp15.c15_power_diagnostic = val;

                }

            default:

                break;

            }

        }

        break;

    }

    return;

bad_reg:

    /* ??? For debugging only.  Should raise illegal instruction exception.  */

    cpu_abort(env, "Unimplemented cp15 register write (c%d, c%d, {%d, %d})\n",

              (insn >> 16) & 0xf, crm, op1, op2);

}
