static void tpm_tis_receive_cb(TPMState *s, uint8_t locty)

{

    TPMTISEmuState *tis = &s->s.tis;



    assert(s->locty_number == locty);



    qemu_bh_schedule(tis->bh);

}
