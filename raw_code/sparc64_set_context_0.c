void sparc64_set_context(CPUSPARCState *env)

{

    abi_ulong ucp_addr;

    struct target_ucontext *ucp;

    target_mc_gregset_t *grp;

    abi_ulong pc, npc, tstate;

    abi_ulong fp, i7, w_addr;

    int err;

    unsigned int i;



    ucp_addr = env->regwptr[UREG_I0];

    if (!lock_user_struct(VERIFY_READ, ucp, ucp_addr, 1))

        goto do_sigsegv;

    grp  = &ucp->tuc_mcontext.mc_gregs;

    err  = __get_user(pc, &((*grp)[MC_PC]));

    err |= __get_user(npc, &((*grp)[MC_NPC]));

    if (err || ((pc | npc) & 3))

        goto do_sigsegv;

    if (env->regwptr[UREG_I1]) {

        target_sigset_t target_set;

        sigset_t set;



        if (TARGET_NSIG_WORDS == 1) {

            if (__get_user(target_set.sig[0], &ucp->tuc_sigmask.sig[0]))

                goto do_sigsegv;

        } else {

            abi_ulong *src, *dst;

            src = ucp->tuc_sigmask.sig;

            dst = target_set.sig;

            for (i = 0; i < TARGET_NSIG_WORDS; i++, dst++, src++) {

                err |= __get_user(*dst, src);

            }

            if (err)

                goto do_sigsegv;

        }

        target_to_host_sigset_internal(&set, &target_set);

        sigprocmask(SIG_SETMASK, &set, NULL);

    }

    env->pc = pc;

    env->npc = npc;

    err |= __get_user(env->y, &((*grp)[MC_Y]));

    err |= __get_user(tstate, &((*grp)[MC_TSTATE]));

    env->asi = (tstate >> 24) & 0xff;

    cpu_put_ccr(env, tstate >> 32);

    cpu_put_cwp64(env, tstate & 0x1f);

    err |= __get_user(env->gregs[1], (&(*grp)[MC_G1]));

    err |= __get_user(env->gregs[2], (&(*grp)[MC_G2]));

    err |= __get_user(env->gregs[3], (&(*grp)[MC_G3]));

    err |= __get_user(env->gregs[4], (&(*grp)[MC_G4]));

    err |= __get_user(env->gregs[5], (&(*grp)[MC_G5]));

    err |= __get_user(env->gregs[6], (&(*grp)[MC_G6]));

    err |= __get_user(env->gregs[7], (&(*grp)[MC_G7]));

    err |= __get_user(env->regwptr[UREG_I0], (&(*grp)[MC_O0]));

    err |= __get_user(env->regwptr[UREG_I1], (&(*grp)[MC_O1]));

    err |= __get_user(env->regwptr[UREG_I2], (&(*grp)[MC_O2]));

    err |= __get_user(env->regwptr[UREG_I3], (&(*grp)[MC_O3]));

    err |= __get_user(env->regwptr[UREG_I4], (&(*grp)[MC_O4]));

    err |= __get_user(env->regwptr[UREG_I5], (&(*grp)[MC_O5]));

    err |= __get_user(env->regwptr[UREG_I6], (&(*grp)[MC_O6]));

    err |= __get_user(env->regwptr[UREG_I7], (&(*grp)[MC_O7]));



    err |= __get_user(fp, &(ucp->tuc_mcontext.mc_fp));

    err |= __get_user(i7, &(ucp->tuc_mcontext.mc_i7));



    w_addr = TARGET_STACK_BIAS+env->regwptr[UREG_I6];

    if (put_user(fp, w_addr + offsetof(struct target_reg_window, ins[6]), 

                 abi_ulong) != 0)

        goto do_sigsegv;

    if (put_user(i7, w_addr + offsetof(struct target_reg_window, ins[7]), 

                 abi_ulong) != 0)

        goto do_sigsegv;

    /* FIXME this does not match how the kernel handles the FPU in

     * its sparc64_set_context implementation. In particular the FPU

     * is only restored if fenab is non-zero in:

     *   __get_user(fenab, &(ucp->tuc_mcontext.mc_fpregs.mcfpu_enab));

     */

    err |= __get_user(env->fprs, &(ucp->tuc_mcontext.mc_fpregs.mcfpu_fprs));

    {

        uint32_t *src = ucp->tuc_mcontext.mc_fpregs.mcfpu_fregs.sregs;

        for (i = 0; i < 64; i++, src++) {

            if (i & 1) {

                err |= __get_user(env->fpr[i/2].l.lower, src);

            } else {

                err |= __get_user(env->fpr[i/2].l.upper, src);

            }

        }

    }

    err |= __get_user(env->fsr,

                      &(ucp->tuc_mcontext.mc_fpregs.mcfpu_fsr));

    err |= __get_user(env->gsr,

                      &(ucp->tuc_mcontext.mc_fpregs.mcfpu_gsr));

    if (err)

        goto do_sigsegv;

    unlock_user_struct(ucp, ucp_addr, 0);

    return;

 do_sigsegv:

    unlock_user_struct(ucp, ucp_addr, 0);

    force_sig(TARGET_SIGSEGV);

}
