static uint64_t mipsnet_ioport_read(void *opaque, target_phys_addr_t addr,

                                    unsigned int size)

{

    MIPSnetState *s = opaque;

    int ret = 0;



    addr &= 0x3f;

    switch (addr) {

    case MIPSNET_DEV_ID:

	ret = be32_to_cpu(0x4d495053);		/* MIPS */

        break;

    case MIPSNET_DEV_ID + 4:

	ret = be32_to_cpu(0x4e455430);		/* NET0 */

        break;

    case MIPSNET_BUSY:

	ret = s->busy;

        break;

    case MIPSNET_RX_DATA_COUNT:

	ret = s->rx_count;

        break;

    case MIPSNET_TX_DATA_COUNT:

	ret = s->tx_count;

        break;

    case MIPSNET_INT_CTL:

	ret = s->intctl;

        s->intctl &= ~MIPSNET_INTCTL_TESTBIT;

        break;

    case MIPSNET_INTERRUPT_INFO:

        /* XXX: This seems to be a per-VPE interrupt number. */

	ret = 0;

        break;

    case MIPSNET_RX_DATA_BUFFER:

        if (s->rx_count) {

            s->rx_count--;

            ret = s->rx_buffer[s->rx_read++];

        }

        break;

    /* Reads as zero. */

    case MIPSNET_TX_DATA_BUFFER:

    default:

        break;

    }

    trace_mipsnet_read(addr, ret);

    return ret;

}
