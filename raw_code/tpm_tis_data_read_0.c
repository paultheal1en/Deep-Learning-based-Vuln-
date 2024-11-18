static uint32_t tpm_tis_data_read(TPMState *s, uint8_t locty)

{

    TPMTISEmuState *tis = &s->s.tis;

    uint32_t ret = TPM_TIS_NO_DATA_BYTE;

    uint16_t len;



    if ((tis->loc[locty].sts & TPM_TIS_STS_DATA_AVAILABLE)) {

        len = tpm_tis_get_size_from_buffer(&tis->loc[locty].r_buffer);



        ret = tis->loc[locty].r_buffer.buffer[tis->loc[locty].r_offset++];

        if (tis->loc[locty].r_offset >= len) {

            /* got last byte */

            tis->loc[locty].sts = TPM_TIS_STS_VALID;

#ifdef RAISE_STS_IRQ

            tpm_tis_raise_irq(s, locty, TPM_TIS_INT_STS_VALID);

#endif

        }

        DPRINTF("tpm_tis: tpm_tis_data_read byte 0x%02x   [%d]\n",

                ret, tis->loc[locty].r_offset-1);

    }



    return ret;

}
