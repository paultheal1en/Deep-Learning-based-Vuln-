static void tpm_tis_receive_bh(void *opaque)

{

    TPMState *s = opaque;

    TPMTISEmuState *tis = &s->s.tis;

    uint8_t locty = s->locty_number;



    tis->loc[locty].sts = TPM_TIS_STS_VALID | TPM_TIS_STS_DATA_AVAILABLE;

    tis->loc[locty].state = TPM_TIS_STATE_COMPLETION;

    tis->loc[locty].r_offset = 0;

    tis->loc[locty].w_offset = 0;



    if (TPM_TIS_IS_VALID_LOCTY(tis->next_locty)) {

        tpm_tis_abort(s, locty);

    }



#ifndef RAISE_STS_IRQ

    tpm_tis_raise_irq(s, locty, TPM_TIS_INT_DATA_AVAILABLE);

#else

    tpm_tis_raise_irq(s, locty,

                      TPM_TIS_INT_DATA_AVAILABLE | TPM_TIS_INT_STS_VALID);

#endif

}
