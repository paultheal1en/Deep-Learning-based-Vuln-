dma_write(void *opaque, target_phys_addr_t addr,

	  uint64_t val64, unsigned int size)

{

        struct fs_dma_ctrl *ctrl = opaque;

	uint32_t value = val64;

	int c;



	if (size != 4) {

		dma_winvalid(opaque, addr, value);

	}



        /* Make addr relative to this channel and bounded to nr regs.  */

	c = fs_channel(addr);

        addr &= 0xff;

        addr >>= 2;

        switch (addr)

	{

		case RW_DATA:

			ctrl->channels[c].regs[addr] = value;

			break;



		case RW_CFG:

			ctrl->channels[c].regs[addr] = value;

			dma_update_state(ctrl, c);

			break;

		case RW_CMD:

			/* continue.  */

			if (value & ~1)

				printf("Invalid store to ch=%d RW_CMD %x\n",

				       c, value);

			ctrl->channels[c].regs[addr] = value;

			channel_continue(ctrl, c);

			break;



		case RW_SAVED_DATA:

		case RW_SAVED_DATA_BUF:

		case RW_GROUP:

		case RW_GROUP_DOWN:

			ctrl->channels[c].regs[addr] = value;

			break;



		case RW_ACK_INTR:

		case RW_INTR_MASK:

			ctrl->channels[c].regs[addr] = value;

			channel_update_irq(ctrl, c);

			if (addr == RW_ACK_INTR)

				ctrl->channels[c].regs[RW_ACK_INTR] = 0;

			break;



		case RW_STREAM_CMD:

			if (value & ~1023)

				printf("Invalid store to ch=%d "

				       "RW_STREAMCMD %x\n",

				       c, value);

			ctrl->channels[c].regs[addr] = value;

			D(printf("stream_cmd ch=%d\n", c));

			channel_stream_cmd(ctrl, c, value);

			break;



	        default:

			D(printf ("%s c=%d " TARGET_FMT_plx "\n",

				__func__, c, addr));

			break;

        }

}
