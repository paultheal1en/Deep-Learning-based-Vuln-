static void pl181_fifo_run(pl181_state *s)

{

    uint32_t bits;

    uint32_t value;

    int n;

    int limit;

    int is_read;



    is_read = (s->datactrl & PL181_DATA_DIRECTION) != 0;

    if (s->datacnt != 0 && (!is_read || sd_data_ready(s->card))

            && !s->linux_hack) {

        limit = is_read ? PL181_FIFO_LEN : 0;

        n = 0;

        value = 0;

        while (s->datacnt && s->fifo_len != limit) {

            if (is_read) {

                value |= (uint32_t)sd_read_data(s->card) << (n * 8);

                n++;

                if (n == 4) {

                    pl181_fifo_push(s, value);

                    value = 0;

                    n = 0;

                }

            } else {

                if (n == 0) {

                    value = pl181_fifo_pop(s);

                    n = 4;

                }

                sd_write_data(s->card, value & 0xff);

                value >>= 8;

                n--;

            }

            s->datacnt--;

        }

        if (n && is_read) {

            pl181_fifo_push(s, value);

        }

    }

    s->status &= ~(PL181_STATUS_RX_FIFO | PL181_STATUS_TX_FIFO);

    if (s->datacnt == 0) {

        s->status |= PL181_STATUS_DATAEND;

        /* HACK: */

        s->status |= PL181_STATUS_DATABLOCKEND;

        DPRINTF("Transfer Complete\n");

    }

    if (s->datacnt == 0 && s->fifo_len == 0) {

        s->datactrl &= ~PL181_DATA_ENABLE;

        DPRINTF("Data engine idle\n");

    } else {

        /* Update FIFO bits.  */

        bits = PL181_STATUS_TXACTIVE | PL181_STATUS_RXACTIVE;

        if (s->fifo_len == 0) {

            bits |= PL181_STATUS_TXFIFOEMPTY;

            bits |= PL181_STATUS_RXFIFOEMPTY;

        } else {

            bits |= PL181_STATUS_TXDATAAVLBL;

            bits |= PL181_STATUS_RXDATAAVLBL;

        }

        if (s->fifo_len == 16) {

            bits |= PL181_STATUS_TXFIFOFULL;

            bits |= PL181_STATUS_RXFIFOFULL;

        }

        if (s->fifo_len <= 8) {

            bits |= PL181_STATUS_TXFIFOHALFEMPTY;

        }

        if (s->fifo_len >= 8) {

            bits |= PL181_STATUS_RXFIFOHALFFULL;

        }

        if (s->datactrl & PL181_DATA_DIRECTION) {

            bits &= PL181_STATUS_RX_FIFO;

        } else {

            bits &= PL181_STATUS_TX_FIFO;

        }

        s->status |= bits;

    }

}
