static void do_v7m_exception_exit(ARMCPU *cpu)

{

    CPUARMState *env = &cpu->env;

    CPUState *cs = CPU(cpu);

    uint32_t excret;

    uint32_t xpsr;

    bool ufault = false;

    bool sfault = false;

    bool return_to_sp_process;

    bool return_to_handler;

    bool rettobase = false;

    bool exc_secure = false;

    bool return_to_secure;



    /* We can only get here from an EXCP_EXCEPTION_EXIT, and

     * gen_bx_excret() enforces the architectural rule

     * that jumps to magic addresses don't have magic behaviour unless

     * we're in Handler mode (compare pseudocode BXWritePC()).

     */

    assert(arm_v7m_is_handler_mode(env));



    /* In the spec pseudocode ExceptionReturn() is called directly

     * from BXWritePC() and gets the full target PC value including

     * bit zero. In QEMU's implementation we treat it as a normal

     * jump-to-register (which is then caught later on), and so split

     * the target value up between env->regs[15] and env->thumb in

     * gen_bx(). Reconstitute it.

     */

    excret = env->regs[15];

    if (env->thumb) {

        excret |= 1;

    }



    qemu_log_mask(CPU_LOG_INT, "Exception return: magic PC %" PRIx32

                  " previous exception %d\n",

                  excret, env->v7m.exception);



    if ((excret & R_V7M_EXCRET_RES1_MASK) != R_V7M_EXCRET_RES1_MASK) {

        qemu_log_mask(LOG_GUEST_ERROR, "M profile: zero high bits in exception "

                      "exit PC value 0x%" PRIx32 " are UNPREDICTABLE\n",

                      excret);

    }



    if (arm_feature(env, ARM_FEATURE_M_SECURITY)) {

        /* EXC_RETURN.ES validation check (R_SMFL). We must do this before

         * we pick which FAULTMASK to clear.

         */

        if (!env->v7m.secure &&

            ((excret & R_V7M_EXCRET_ES_MASK) ||

             !(excret & R_V7M_EXCRET_DCRS_MASK))) {

            sfault = 1;

            /* For all other purposes, treat ES as 0 (R_HXSR) */

            excret &= ~R_V7M_EXCRET_ES_MASK;

        }

    }



    if (env->v7m.exception != ARMV7M_EXCP_NMI) {

        /* Auto-clear FAULTMASK on return from other than NMI.

         * If the security extension is implemented then this only

         * happens if the raw execution priority is >= 0; the

         * value of the ES bit in the exception return value indicates

         * which security state's faultmask to clear. (v8M ARM ARM R_KBNF.)

         */

        if (arm_feature(env, ARM_FEATURE_M_SECURITY)) {

            exc_secure = excret & R_V7M_EXCRET_ES_MASK;

            if (armv7m_nvic_raw_execution_priority(env->nvic) >= 0) {

                env->v7m.faultmask[exc_secure] = 0;

            }

        } else {

            env->v7m.faultmask[M_REG_NS] = 0;

        }

    }



    switch (armv7m_nvic_complete_irq(env->nvic, env->v7m.exception,

                                     exc_secure)) {

    case -1:

        /* attempt to exit an exception that isn't active */

        ufault = true;

        break;

    case 0:

        /* still an irq active now */

        break;

    case 1:

        /* we returned to base exception level, no nesting.

         * (In the pseudocode this is written using "NestedActivation != 1"

         * where we have 'rettobase == false'.)

         */

        rettobase = true;

        break;

    default:

        g_assert_not_reached();

    }



    return_to_handler = !(excret & R_V7M_EXCRET_MODE_MASK);

    return_to_sp_process = excret & R_V7M_EXCRET_SPSEL_MASK;

    return_to_secure = arm_feature(env, ARM_FEATURE_M_SECURITY) &&

        (excret & R_V7M_EXCRET_S_MASK);



    if (arm_feature(env, ARM_FEATURE_V8)) {

        if (!arm_feature(env, ARM_FEATURE_M_SECURITY)) {

            /* UNPREDICTABLE if S == 1 or DCRS == 0 or ES == 1 (R_XLCP);

             * we choose to take the UsageFault.

             */

            if ((excret & R_V7M_EXCRET_S_MASK) ||

                (excret & R_V7M_EXCRET_ES_MASK) ||

                !(excret & R_V7M_EXCRET_DCRS_MASK)) {

                ufault = true;

            }

        }

        if (excret & R_V7M_EXCRET_RES0_MASK) {

            ufault = true;

        }

    } else {

        /* For v7M we only recognize certain combinations of the low bits */

        switch (excret & 0xf) {

        case 1: /* Return to Handler */

            break;

        case 13: /* Return to Thread using Process stack */

        case 9: /* Return to Thread using Main stack */

            /* We only need to check NONBASETHRDENA for v7M, because in

             * v8M this bit does not exist (it is RES1).

             */

            if (!rettobase &&

                !(env->v7m.ccr[env->v7m.secure] &

                  R_V7M_CCR_NONBASETHRDENA_MASK)) {

                ufault = true;

            }

            break;

        default:

            ufault = true;

        }

    }



    if (sfault) {

        env->v7m.sfsr |= R_V7M_SFSR_INVER_MASK;

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_SECURE, false);

        v7m_exception_taken(cpu, excret);

        qemu_log_mask(CPU_LOG_INT, "...taking SecureFault on existing "

                      "stackframe: failed EXC_RETURN.ES validity check\n");

        return;

    }



    if (ufault) {

        /* Bad exception return: instead of popping the exception

         * stack, directly take a usage fault on the current stack.

         */

        env->v7m.cfsr[env->v7m.secure] |= R_V7M_CFSR_INVPC_MASK;

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE, env->v7m.secure);

        v7m_exception_taken(cpu, excret);

        qemu_log_mask(CPU_LOG_INT, "...taking UsageFault on existing "

                      "stackframe: failed exception return integrity check\n");

        return;

    }



    /* Set CONTROL.SPSEL from excret.SPSEL. Since we're still in

     * Handler mode (and will be until we write the new XPSR.Interrupt

     * field) this does not switch around the current stack pointer.

     */

    write_v7m_control_spsel_for_secstate(env, return_to_sp_process, exc_secure);



    switch_v7m_security_state(env, return_to_secure);



    {

        /* The stack pointer we should be reading the exception frame from

         * depends on bits in the magic exception return type value (and

         * for v8M isn't necessarily the stack pointer we will eventually

         * end up resuming execution with). Get a pointer to the location

         * in the CPU state struct where the SP we need is currently being

         * stored; we will use and modify it in place.

         * We use this limited C variable scope so we don't accidentally

         * use 'frame_sp_p' after we do something that makes it invalid.

         */

        uint32_t *frame_sp_p = get_v7m_sp_ptr(env,

                                              return_to_secure,

                                              !return_to_handler,

                                              return_to_sp_process);

        uint32_t frameptr = *frame_sp_p;



        if (!QEMU_IS_ALIGNED(frameptr, 8) &&

            arm_feature(env, ARM_FEATURE_V8)) {

            qemu_log_mask(LOG_GUEST_ERROR,

                          "M profile exception return with non-8-aligned SP "

                          "for destination state is UNPREDICTABLE\n");

        }



        /* Do we need to pop callee-saved registers? */

        if (return_to_secure &&

            ((excret & R_V7M_EXCRET_ES_MASK) == 0 ||

             (excret & R_V7M_EXCRET_DCRS_MASK) == 0)) {

            uint32_t expected_sig = 0xfefa125b;

            uint32_t actual_sig = ldl_phys(cs->as, frameptr);



            if (expected_sig != actual_sig) {

                /* Take a SecureFault on the current stack */

                env->v7m.sfsr |= R_V7M_SFSR_INVIS_MASK;

                armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_SECURE, false);

                v7m_exception_taken(cpu, excret);

                qemu_log_mask(CPU_LOG_INT, "...taking SecureFault on existing "

                              "stackframe: failed exception return integrity "

                              "signature check\n");

                return;

            }



            env->regs[4] = ldl_phys(cs->as, frameptr + 0x8);

            env->regs[5] = ldl_phys(cs->as, frameptr + 0xc);

            env->regs[6] = ldl_phys(cs->as, frameptr + 0x10);

            env->regs[7] = ldl_phys(cs->as, frameptr + 0x14);

            env->regs[8] = ldl_phys(cs->as, frameptr + 0x18);

            env->regs[9] = ldl_phys(cs->as, frameptr + 0x1c);

            env->regs[10] = ldl_phys(cs->as, frameptr + 0x20);

            env->regs[11] = ldl_phys(cs->as, frameptr + 0x24);



            frameptr += 0x28;

        }



        /* Pop registers. TODO: make these accesses use the correct

         * attributes and address space (S/NS, priv/unpriv) and handle

         * memory transaction failures.

         */

        env->regs[0] = ldl_phys(cs->as, frameptr);

        env->regs[1] = ldl_phys(cs->as, frameptr + 0x4);

        env->regs[2] = ldl_phys(cs->as, frameptr + 0x8);

        env->regs[3] = ldl_phys(cs->as, frameptr + 0xc);

        env->regs[12] = ldl_phys(cs->as, frameptr + 0x10);

        env->regs[14] = ldl_phys(cs->as, frameptr + 0x14);

        env->regs[15] = ldl_phys(cs->as, frameptr + 0x18);



        /* Returning from an exception with a PC with bit 0 set is defined

         * behaviour on v8M (bit 0 is ignored), but for v7M it was specified

         * to be UNPREDICTABLE. In practice actual v7M hardware seems to ignore

         * the lsbit, and there are several RTOSes out there which incorrectly

         * assume the r15 in the stack frame should be a Thumb-style "lsbit

         * indicates ARM/Thumb" value, so ignore the bit on v7M as well, but

         * complain about the badly behaved guest.

         */

        if (env->regs[15] & 1) {

            env->regs[15] &= ~1U;

            if (!arm_feature(env, ARM_FEATURE_V8)) {

                qemu_log_mask(LOG_GUEST_ERROR,

                              "M profile return from interrupt with misaligned "

                              "PC is UNPREDICTABLE on v7M\n");

            }

        }



        xpsr = ldl_phys(cs->as, frameptr + 0x1c);



        if (arm_feature(env, ARM_FEATURE_V8)) {

            /* For v8M we have to check whether the xPSR exception field

             * matches the EXCRET value for return to handler/thread

             * before we commit to changing the SP and xPSR.

             */

            bool will_be_handler = (xpsr & XPSR_EXCP) != 0;

            if (return_to_handler != will_be_handler) {

                /* Take an INVPC UsageFault on the current stack.

                 * By this point we will have switched to the security state

                 * for the background state, so this UsageFault will target

                 * that state.

                 */

                armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE,

                                        env->v7m.secure);

                env->v7m.cfsr[env->v7m.secure] |= R_V7M_CFSR_INVPC_MASK;

                v7m_exception_taken(cpu, excret);

                qemu_log_mask(CPU_LOG_INT, "...taking UsageFault on existing "

                              "stackframe: failed exception return integrity "

                              "check\n");

                return;

            }

        }



        /* Commit to consuming the stack frame */

        frameptr += 0x20;

        /* Undo stack alignment (the SPREALIGN bit indicates that the original

         * pre-exception SP was not 8-aligned and we added a padding word to

         * align it, so we undo this by ORing in the bit that increases it

         * from the current 8-aligned value to the 8-unaligned value. (Adding 4

         * would work too but a logical OR is how the pseudocode specifies it.)

         */

        if (xpsr & XPSR_SPREALIGN) {

            frameptr |= 4;

        }

        *frame_sp_p = frameptr;

    }

    /* This xpsr_write() will invalidate frame_sp_p as it may switch stack */

    xpsr_write(env, xpsr, ~XPSR_SPREALIGN);



    /* The restored xPSR exception field will be zero if we're

     * resuming in Thread mode. If that doesn't match what the

     * exception return excret specified then this is a UsageFault.

     * v7M requires we make this check here; v8M did it earlier.

     */

    if (return_to_handler != arm_v7m_is_handler_mode(env)) {

        /* Take an INVPC UsageFault by pushing the stack again;

         * we know we're v7M so this is never a Secure UsageFault.

         */

        assert(!arm_feature(env, ARM_FEATURE_V8));

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE, false);

        env->v7m.cfsr[env->v7m.secure] |= R_V7M_CFSR_INVPC_MASK;

        v7m_push_stack(cpu);

        v7m_exception_taken(cpu, excret);

        qemu_log_mask(CPU_LOG_INT, "...taking UsageFault on new stackframe: "

                      "failed exception return integrity check\n");

        return;

    }



    /* Otherwise, we have a successful exception exit. */

    arm_clear_exclusive(env);

    qemu_log_mask(CPU_LOG_INT, "...successful exception return\n");

}
