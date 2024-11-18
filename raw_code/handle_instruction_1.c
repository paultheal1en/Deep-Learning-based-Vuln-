static int handle_instruction(CPUState *env, struct kvm_run *run)

{

    unsigned int ipa0 = (run->s390_sieic.ipa & 0xff00);

    uint8_t ipa1 = run->s390_sieic.ipa & 0x00ff;

    int ipb_code = (run->s390_sieic.ipb & 0x0fff0000) >> 16;

    int r = -1;



    dprintf("handle_instruction 0x%x 0x%x\n", run->s390_sieic.ipa, run->s390_sieic.ipb);

    switch (ipa0) {

        case IPA0_PRIV:

            r = handle_priv(env, run, ipa1);

            break;

        case IPA0_DIAG:

            r = handle_diag(env, run, ipb_code);

            break;

        case IPA0_SIGP:

            r = handle_sigp(env, run, ipa1);

            break;

    }



    if (r < 0) {

        enter_pgmcheck(env, 0x0001);

    }

    return r;

}
