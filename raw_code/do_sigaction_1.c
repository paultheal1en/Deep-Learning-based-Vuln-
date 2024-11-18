int do_sigaction(int sig, const struct target_sigaction *act,

                 struct target_sigaction *oact)

{

    struct emulated_sigaction *k;



    if (sig < 1 || sig > TARGET_NSIG)

        return -EINVAL;

    k = &sigact_table[sig - 1];

#if defined(DEBUG_SIGNAL) && 0

    fprintf(stderr, "sigaction sig=%d act=0x%08x, oact=0x%08x\n", 

            sig, (int)act, (int)oact);

#endif

    if (oact) {

        oact->_sa_handler = tswapl(k->sa._sa_handler);

        oact->sa_flags = tswapl(k->sa.sa_flags);

        oact->sa_restorer = tswapl(k->sa.sa_restorer);

        oact->sa_mask = k->sa.sa_mask;

    }

    if (act) {

        k->sa._sa_handler = tswapl(act->_sa_handler);

        k->sa.sa_flags = tswapl(act->sa_flags);

        k->sa.sa_restorer = tswapl(act->sa_restorer);

        k->sa.sa_mask = act->sa_mask;

    }

    return 0;

}
