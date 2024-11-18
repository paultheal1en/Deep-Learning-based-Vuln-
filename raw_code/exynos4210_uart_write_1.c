static void exynos4210_uart_write(void *opaque, hwaddr offset,

                               uint64_t val, unsigned size)

{

    Exynos4210UartState *s = (Exynos4210UartState *)opaque;

    uint8_t ch;



    PRINT_DEBUG_EXTEND("UART%d: <0x%04x> %s <- 0x%08llx\n", s->channel,

        offset, exynos4210_uart_regname(offset), (long long unsigned int)val);



    switch (offset) {

    case ULCON:

    case UBRDIV:

    case UFRACVAL:

        s->reg[I_(offset)] = val;

        exynos4210_uart_update_parameters(s);

        break;

    case UFCON:

        s->reg[I_(UFCON)] = val;

        if (val & UFCON_Rx_FIFO_RESET) {

            fifo_reset(&s->rx);

            s->reg[I_(UFCON)] &= ~UFCON_Rx_FIFO_RESET;

            PRINT_DEBUG("UART%d: Rx FIFO Reset\n", s->channel);

        }

        if (val & UFCON_Tx_FIFO_RESET) {

            fifo_reset(&s->tx);

            s->reg[I_(UFCON)] &= ~UFCON_Tx_FIFO_RESET;

            PRINT_DEBUG("UART%d: Tx FIFO Reset\n", s->channel);

        }

        break;



    case UTXH:

        if (s->chr) {

            s->reg[I_(UTRSTAT)] &= ~(UTRSTAT_TRANSMITTER_EMPTY |

                    UTRSTAT_Tx_BUFFER_EMPTY);

            ch = (uint8_t)val;

            qemu_chr_fe_write(s->chr, &ch, 1);

#if DEBUG_Tx_DATA

            fprintf(stderr, "%c", ch);

#endif

            s->reg[I_(UTRSTAT)] |= UTRSTAT_TRANSMITTER_EMPTY |

                    UTRSTAT_Tx_BUFFER_EMPTY;

            s->reg[I_(UINTSP)]  |= UINTSP_TXD;

            exynos4210_uart_update_irq(s);

        }

        break;



    case UINTP:

        s->reg[I_(UINTP)] &= ~val;

        s->reg[I_(UINTSP)] &= ~val;

        PRINT_DEBUG("UART%d: UINTP [%04x] have been cleared: %08x\n",

                    s->channel, offset, s->reg[I_(UINTP)]);

        exynos4210_uart_update_irq(s);

        break;

    case UTRSTAT:

    case UERSTAT:

    case UFSTAT:

    case UMSTAT:

    case URXH:

        PRINT_DEBUG("UART%d: Trying to write into RO register: %s [%04x]\n",

                    s->channel, exynos4210_uart_regname(offset), offset);

        break;

    case UINTSP:

        s->reg[I_(UINTSP)]  &= ~val;

        break;

    case UINTM:

        s->reg[I_(UINTM)] = val;

        exynos4210_uart_update_irq(s);

        break;

    case UCON:

    case UMCON:

    default:

        s->reg[I_(offset)] = val;

        break;

    }

}
