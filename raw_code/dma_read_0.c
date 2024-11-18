dma_read(void *opaque, target_phys_addr_t addr, unsigned int size)

{

        struct fs_dma_ctrl *ctrl = opaque;

	int c;

	uint32_t r = 0;



	if (size != 4) {

		dma_rinvalid(opaque, addr);

	}



	/* Make addr relative to this channel and bounded to nr regs.  */

	c = fs_channel(addr);

	addr &= 0xff;

	addr >>= 2;

	switch (addr)

	{

		case RW_STAT:

			r = ctrl->channels[c].state & 7;

			r |= ctrl->channels[c].eol << 5;

			r |= ctrl->channels[c].stream_cmd_src << 8;

			break;



		default:

			r = ctrl->channels[c].regs[addr];

			D(printf ("%s c=%d addr=" TARGET_FMT_plx "\n",

				  __func__, c, addr));

			break;

	}

	return r;

}
