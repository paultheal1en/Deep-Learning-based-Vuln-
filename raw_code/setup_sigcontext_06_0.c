static int setup_sigcontext(struct target_sigcontext *sc,

                            CPUOpenRISCState *regs,

                            unsigned long mask)

{

    int err = 0;

    unsigned long usp = regs->gpr[1];



    /* copy the regs. they are first in sc so we can use sc directly */



    /*copy_to_user(&sc, regs, sizeof(struct target_pt_regs));*/



    /* Set the frametype to CRIS_FRAME_NORMAL for the execution of

       the signal handler. The frametype will be restored to its previous

       value in restore_sigcontext. */

    /*regs->frametype = CRIS_FRAME_NORMAL;*/



    /* then some other stuff */

    __put_user(mask, &sc->oldmask);

    __put_user(usp, &sc->usp); return err;

}
