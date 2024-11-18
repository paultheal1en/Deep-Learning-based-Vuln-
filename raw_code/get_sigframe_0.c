static target_ulong get_sigframe(struct target_sigaction *ka,

                                 CPUPPCState *env,

                                 int frame_size)

{

    target_ulong oldsp, newsp;



    oldsp = env->gpr[1];



    if ((ka->sa_flags & TARGET_SA_ONSTACK) &&

        (sas_ss_flags(oldsp))) {

        oldsp = (target_sigaltstack_used.ss_sp

                 + target_sigaltstack_used.ss_size);

    }



    newsp = (oldsp - frame_size) & ~0xFUL;



    return newsp;

}
