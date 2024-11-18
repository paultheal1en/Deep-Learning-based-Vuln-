static int setup_sigcontext(struct target_sigcontext *sc,

                            CPUSH4State *regs, unsigned long mask)

{

    int err = 0;

    int i;



#define COPY(x)         __put_user(regs->x, &sc->sc_##x)

    COPY(gregs[0]); COPY(gregs[1]);

    COPY(gregs[2]); COPY(gregs[3]);

    COPY(gregs[4]); COPY(gregs[5]);

    COPY(gregs[6]); COPY(gregs[7]);

    COPY(gregs[8]); COPY(gregs[9]);

    COPY(gregs[10]); COPY(gregs[11]);

    COPY(gregs[12]); COPY(gregs[13]);

    COPY(gregs[14]); COPY(gregs[15]);

    COPY(gbr); COPY(mach);

    COPY(macl); COPY(pr);

    COPY(sr); COPY(pc);

#undef COPY



    for (i=0; i<16; i++) {

        __put_user(regs->fregs[i], &sc->sc_fpregs[i]);

    }

    __put_user(regs->fpscr, &sc->sc_fpscr);

    __put_user(regs->fpul, &sc->sc_fpul);



    /* non-iBCS2 extensions.. */

    __put_user(mask, &sc->oldmask);



    return err;

}
