eth_write(void *opaque, target_phys_addr_t addr,

          uint64_t val64, unsigned int size)

{

	struct fs_eth *eth = opaque;

	uint32_t value = val64;



	addr >>= 2;

	switch (addr)

	{

		case RW_MA0_LO:

		case RW_MA0_HI:

			eth->regs[addr] = value;

			eth_update_ma(eth, 0);

			break;

		case RW_MA1_LO:

		case RW_MA1_HI:

			eth->regs[addr] = value;

			eth_update_ma(eth, 1);

			break;



		case RW_MGM_CTRL:

			/* Attach an MDIO/PHY abstraction.  */

			if (value & 2)

				eth->mdio_bus.mdio = value & 1;

			if (eth->mdio_bus.mdc != (value & 4)) {

				mdio_cycle(&eth->mdio_bus);

				eth_validate_duplex(eth);

			}

			eth->mdio_bus.mdc = !!(value & 4);

			eth->regs[addr] = value;

			break;



		case RW_REC_CTRL:

			eth->regs[addr] = value;

			eth_validate_duplex(eth);

			break;



		default:

			eth->regs[addr] = value;

			D(printf ("%s %x %x\n",

				  __func__, addr, value));

			break;

	}

}
