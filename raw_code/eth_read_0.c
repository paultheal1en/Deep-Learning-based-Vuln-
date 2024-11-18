eth_read(void *opaque, target_phys_addr_t addr, unsigned int size)

{

	struct fs_eth *eth = opaque;

	uint32_t r = 0;



	addr >>= 2;



	switch (addr) {

		case R_STAT:

			r = eth->mdio_bus.mdio & 1;

			break;

	default:

		r = eth->regs[addr];

		D(printf ("%s %x\n", __func__, addr * 4));

		break;

	}

	return r;

}
