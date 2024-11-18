int has_altivec(void)

{

#ifdef __AMIGAOS4__

    ULONG result = 0;

    extern struct ExecIFace *IExec;



    IExec->GetCPUInfoTags(GCIT_VectorUnit, &result, TAG_DONE);

    if (result == VECTORTYPE_ALTIVEC) return 1;

    return 0;

#else /* __AMIGAOS4__ */



#ifdef SYS_DARWIN

    int sels[2] = {CTL_HW, HW_VECTORUNIT};

    int has_vu = 0;

    size_t len = sizeof(has_vu);

    int err;



    err = sysctl(sels, 2, &has_vu, &len, NULL, 0);



    if (err == 0) return (has_vu != 0);

#else /* SYS_DARWIN */

/* no Darwin, do it the brute-force way */

/* this is borrowed from the libmpeg2 library */

    {

      signal (SIGILL, sigill_handler);

      if (sigsetjmp (jmpbuf, 1)) {

        signal (SIGILL, SIG_DFL);

      } else {

        canjump = 1;



        asm volatile ("mtspr 256, %0\n\t"

                      "vand %%v0, %%v0, %%v0"

                      :

                      : "r" (-1));



        signal (SIGILL, SIG_DFL);

        return 1;

      }

    }

#endif /* SYS_DARWIN */

    return 0;

#endif /* __AMIGAOS4__ */

}
