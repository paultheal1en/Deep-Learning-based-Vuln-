static void sdhci_write_dataport(SDHCIState *s, uint32_t value, unsigned size)

{

    unsigned i;



    /* Check that there is free space left in a buffer */

    if (!(s->prnsts & SDHC_SPACE_AVAILABLE)) {

        ERRPRINT("Can't write to data buffer: buffer full\n");

        return;

    }



    for (i = 0; i < size; i++) {

        s->fifo_buffer[s->data_count] = value & 0xFF;

        s->data_count++;

        value >>= 8;

        if (s->data_count >= (s->blksize & 0x0fff)) {

            DPRINT_L2("write buffer filled with %u bytes of data\n",

                    s->data_count);

            s->data_count = 0;

            s->prnsts &= ~SDHC_SPACE_AVAILABLE;

            if (s->prnsts & SDHC_DOING_WRITE) {

                sdhci_write_block_to_card(s);

            }

        }

    }

}
