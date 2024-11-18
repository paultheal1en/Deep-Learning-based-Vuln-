void handle_diag_308(CPUS390XState *env, uint64_t r1, uint64_t r3)

{

    uint64_t addr =  env->regs[r1];

    uint64_t subcode = env->regs[r3];

    IplParameterBlock *iplb;



    if (env->psw.mask & PSW_MASK_PSTATE) {

        program_interrupt(env, PGM_PRIVILEGED, ILEN_LATER_INC);

        return;

    }



    if ((subcode & ~0x0ffffULL) || (subcode > 6)) {

        program_interrupt(env, PGM_SPECIFICATION, ILEN_LATER_INC);

        return;

    }



    switch (subcode) {

    case 0:

        modified_clear_reset(s390_env_get_cpu(env));

        if (tcg_enabled()) {

            cpu_loop_exit(CPU(s390_env_get_cpu(env)));

        }

        break;

    case 1:

        load_normal_reset(s390_env_get_cpu(env));

        if (tcg_enabled()) {

            cpu_loop_exit(CPU(s390_env_get_cpu(env)));

        }

        break;

    case 3:

        s390_reipl_request();

        if (tcg_enabled()) {

            cpu_loop_exit(CPU(s390_env_get_cpu(env)));

        }

        break;

    case 5:

        if ((r1 & 1) || (addr & 0x0fffULL)) {

            program_interrupt(env, PGM_SPECIFICATION, ILEN_LATER_INC);

            return;

        }

        if (!address_space_access_valid(&address_space_memory, addr,

                                        sizeof(IplParameterBlock), false)) {

            program_interrupt(env, PGM_ADDRESSING, ILEN_LATER_INC);

            return;

        }

        iplb = g_malloc0(sizeof(IplParameterBlock));

        cpu_physical_memory_read(addr, iplb, sizeof(iplb->len));

        if (!iplb_valid_len(iplb)) {

            env->regs[r1 + 1] = DIAG_308_RC_INVALID;

            goto out;

        }



        cpu_physical_memory_read(addr, iplb, be32_to_cpu(iplb->len));



        if (!iplb_valid_ccw(iplb) && !iplb_valid_fcp(iplb)) {

            env->regs[r1 + 1] = DIAG_308_RC_INVALID;

            goto out;

        }



        s390_ipl_update_diag308(iplb);

        env->regs[r1 + 1] = DIAG_308_RC_OK;

out:

        g_free(iplb);

        return;

    case 6:

        if ((r1 & 1) || (addr & 0x0fffULL)) {

            program_interrupt(env, PGM_SPECIFICATION, ILEN_LATER_INC);

            return;

        }

        if (!address_space_access_valid(&address_space_memory, addr,

                                        sizeof(IplParameterBlock), true)) {

            program_interrupt(env, PGM_ADDRESSING, ILEN_LATER_INC);

            return;

        }

        iplb = s390_ipl_get_iplb();

        if (iplb) {

            cpu_physical_memory_write(addr, iplb, be32_to_cpu(iplb->len));

            env->regs[r1 + 1] = DIAG_308_RC_OK;

        } else {

            env->regs[r1 + 1] = DIAG_308_RC_NO_CONF;

        }

        return;

    default:

        hw_error("Unhandled diag308 subcode %" PRIx64, subcode);

        break;

    }

}
